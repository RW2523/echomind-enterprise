import React, { useRef, useEffect, useCallback, useState, useMemo } from "react";
import { getWordCounts, type WordCount } from "../utils/wordCloudUtils";
import { listTranscripts, getTranscript } from "../services/backend";
import { ICONS } from "../constants";

const CLOUD_UPDATE_INTERVAL_MS = 60 * 1000; // 1 minute when live
const MIN_FONT = 14;
const MAX_FONT = 80;
const PADDING = 4;

/** Font scale: log n — size proportional to log(1+count). */
const FONT_SCALE = "log" as "sqrt" | "log" | "linear";

interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

function boxesOverlap(a: Box, b: Box): boolean {
  return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
}

/**
 * Word cloud layout following the Jason Davies / d3-cloud concept:
 * - Place words by importance (highest count first).
 * - Font size scales by frequency: log n so bigger = more frequent.
 * - Archimedean spiral: try center first, then move one step along spiral until no collision.
 * @see https://www.jasondavies.com/wordcloud/
 */
function drawWordCloud(
  canvas: HTMLCanvasElement,
  words: WordCount[],
  width: number,
  height: number
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || words.length === 0) return;

  const centerX = width / 2;
  const centerY = height / 2;
  const placed: { box: Box; word: string; count: number; font: number }[] = [];
  const maxCount = Math.max(...words.map((w) => w.count), 1);

  // Scale font by count (log n)
  function fontForCount(count: number): number {
    let t: number;
    if (FONT_SCALE === "sqrt") {
      t = Math.sqrt(count) / Math.sqrt(maxCount);
    } else if (FONT_SCALE === "log") {
      t = Math.log(1 + count) / Math.log(1 + maxCount);
    } else {
      t = count / maxCount;
    }
    return Math.round(MIN_FONT + (MAX_FONT - MIN_FONT) * Math.min(1, t));
  }

  // Archimedean spiral: r = growth * angle (one step = small angle increment)
  const spiralStep = 0.25;
  const spiralGrowth = 3;
  const maxRadius = Math.sqrt(width * width + height * height) / 2;

  const colors = [
    "#22d3ee", "#38bdf8", "#818cf8", "#a78bfa", "#c084fc",
    "#34d399", "#4ade80", "#fbbf24", "#fb923c", "#f472b6",
  ];

  let spiralAngle = 0; // shared spiral position for placement attempts

  for (let i = 0; i < words.length; i++) {
    const { word, count } = words[i];
    const font = fontForCount(count);
    ctx.font = `${font}px sans-serif`;
    const metrics = ctx.measureText(word);
    const w = metrics.width + PADDING * 2;
    const h = font + PADDING * 2;

    let placedWord = false;
    let tries = 0;
    const maxTries = Math.ceil((maxRadius / spiralGrowth / spiralStep) * 2);

    while (!placedWord && tries < maxTries) {
      const r = spiralAngle * spiralGrowth;
      const x = centerX + Math.cos(spiralAngle) * r - w / 2;
      const y = centerY + Math.sin(spiralAngle) * r - h / 2;
      const left = Math.max(0, Math.min(x, width - w));
      const top = Math.max(0, Math.min(y, height - h));
      const box: Box = { left, top, right: left + w, bottom: top + h };

      const overlaps = placed.some((p) => boxesOverlap(p.box, box));
      if (!overlaps && r <= maxRadius) {
        placed.push({ box, word, count, font });
        placedWord = true;
        spiralAngle += 0.4; // advance so next word doesn’t start at same spot
      }
      spiralAngle += spiralStep;
      tries++;
    }

    if (!placedWord) {
      // Fallback: grid at bottom with MIN_FONT
      const fallbackFont = MIN_FONT;
      ctx.font = `${fallbackFont}px sans-serif`;
      const fw = ctx.measureText(word).width + PADDING * 2;
      const fh = fallbackFont + PADDING * 2;
      let rowY = height - fh - PADDING;
      let gx = PADDING;
      let found = false;
      while (rowY >= PADDING && !found) {
        while (gx + fw <= width - PADDING) {
          const fbox: Box = { left: gx, top: rowY, right: gx + fw, bottom: rowY + fh };
          if (!placed.some((p) => boxesOverlap(p.box, fbox))) {
            placed.push({ box: fbox, word, count, font: fallbackFont });
            found = true;
            break;
          }
          gx += fw;
        }
        gx = PADDING;
        rowY -= fh;
      }
    }
  }

  ctx.clearRect(0, 0, width, height);
  placed.forEach(({ box, word, font }, i) => {
    ctx.fillStyle = colors[i % colors.length];
    ctx.font = `${font}px sans-serif`;
    ctx.textBaseline = "top";
    ctx.fillText(word, box.left + PADDING, box.top + PADDING);
  });
}

interface WordCloudModalProps {
  onClose: () => void;
  liveText: string;
  listening: boolean;
}

const WordCloudModal: React.FC<WordCloudModalProps> = ({ onClose, liveText, listening }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dbText, setDbText] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Memoize the (expensive) tokenization so it only recomputes when the text changes — not on
  // every render / unrelated state update. (M24)
  const words: WordCount[] = useMemo(() => {
    const combinedText = `${dbText}\n${liveText}`.trim();
    const rawWords = getWordCounts(combinedText);
    const topCount = rawWords[0]?.count ?? 0;
    return [{ word: "EchoMind", count: topCount + 10 }, ...rawWords].slice(0, 50);
  }, [dbText, liveText]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || words.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.floor(rect.width * dpr);
    const h = Math.floor(rect.height * dpr);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.scale(dpr, dpr);
    drawWordCloud(canvas, words, rect.width, rect.height);
  }, [words]);

  // Fetch existing transcripts from DB on mount
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const { transcripts } = await listTranscripts();
        // Fetch transcript bodies in parallel instead of one sequential round-trip each (N+1). (M23)
        const details = await Promise.all(
          transcripts.map((t) =>
            getTranscript(t.id).then((d) => d.raw_text || "").catch(() => "")
          )
        );
        if (!cancelled) setDbText(details.filter(Boolean).join("\n"));
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Redraw when words or size change
  useEffect(() => {
    if (loading) return;
    redraw();
  }, [words, loading, redraw]);

  // When live, refresh cloud every 1–2 minutes
  useEffect(() => {
    if (!listening) return;
    const id = setInterval(redraw, CLOUD_UPDATE_INTERVAL_MS);
    return () => clearInterval(id);
  }, [listening, redraw]);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      if (words.length > 0) redraw();
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [words.length, redraw]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-950" onClick={onClose}>
      <div
        className="flex flex-col w-full h-full"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="shrink-0 flex items-center justify-between px-5 py-3 border-b border-white/10 bg-slate-900/80 backdrop-blur">
          <div className="font-semibold text-white flex flex-wrap items-center gap-2">
            <span>Word Cloud</span>
            <span className="text-xs font-normal text-slate-400">
              All previous transcripts + live transcript
            </span>
            {listening && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/40 px-2 py-0.5 text-[10px] font-medium text-cyan-400">
                Live — updates every 1 min
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Close"
          >
            <ICONS.Close className="w-5 h-5" />
          </button>
        </div>

        {/* Full-screen canvas area */}
        <div className="flex-1 min-h-0 p-0 flex flex-col">
          {loading ? (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-lg">
              Loading transcripts…
            </div>
          ) : error ? (
            <div className="flex-1 flex items-center justify-center text-red-400">{error}</div>
          ) : words.length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-lg">
              No words yet. Add transcripts or start a live transcript.
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              className="w-full flex-1 min-h-0 bg-slate-950"
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default WordCloudModal;
