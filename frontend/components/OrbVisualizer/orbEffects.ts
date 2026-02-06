/**
 * Reusable canvas drawing helpers for the orb visualizer.
 * All coordinates in canvas pixel space; center and radius passed in.
 */

export function drawCenterAvatar(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  avatarImage: HTMLImageElement | null,
  role: "user" | "assistant",
  fallbackColor: string
): void {
  const r = radius * 0.72;
  ctx.save();
  ctx.beginPath();
  ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();

  if (avatarImage && avatarImage.complete && avatarImage.naturalWidth > 0) {
    const size = r * 2;
    ctx.drawImage(avatarImage, centerX - r, centerY - r, size, size);
  } else {
    ctx.fillStyle = fallbackColor;
    ctx.fill();
  }
  ctx.restore();
}

export function drawGlowRing(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  intensity: number,
  color: string,
  thickness: number = 4
): void {
  const gradient = ctx.createRadialGradient(
    centerX, centerY, radius * 0.6,
    centerX, centerY, radius * 1.4
  );
  const [r, g, b] = hexToRgb(color);
  gradient.addColorStop(0, `rgba(${r},${g},${b},${intensity * 0.3})`);
  gradient.addColorStop(0.5, `rgba(${r},${g},${b},${intensity * 0.15})`);
  gradient.addColorStop(1, "transparent");
  ctx.save();
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius * 1.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = `rgba(${r},${g},${b},${intensity * 0.6})`;
  ctx.lineWidth = thickness;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function hexToRgb(hex: string): [number, number, number] {
  const m = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [255, 255, 255];
}

export function drawWaveRing(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  timeDomainData: Float32Array,
  amplitude: number,
  color: string,
  _smoothing: number = 0.3
): void {
  if (amplitude <= 0 || timeDomainData.length < 2) return;
  const [r, g, b] = hexToRgb(color);
  const segments = 80;
  const waveScale = 0.22;

  ctx.save();
  ctx.strokeStyle = `rgba(${r},${g},${b},0.45)`;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const sampleIdx = Math.floor(t * (timeDomainData.length - 1));
    const sample = timeDomainData[sampleIdx] ?? 0;
    const prevIdx = Math.max(0, sampleIdx - 8);
    const nextIdx = Math.min(timeDomainData.length - 1, sampleIdx + 8);
    const smooth = (timeDomainData[prevIdx] + sample * 2 + timeDomainData[nextIdx]) / 4;
    const rOff = radius * amplitude * smooth * waveScale;
    const angle = t * Math.PI * 2 - Math.PI / 2;
    const x = centerX + (radius + rOff) * Math.cos(angle);
    const y = centerY + (radius + rOff) * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

export function drawOrbitingParticles(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  count: number,
  time: number,
  color: string,
  state: string
): void {
  if (count <= 0) return;
  const [r, g, b] = hexToRgb(color);
  const orbitRadius = radius * 1.08;
  const particleRadius = Math.max(1, radius * 0.025);
  const isActive = state === "speaking" || state === "listening";
  const speed = isActive ? (state === "speaking" ? 0.5 : 0.4) : 0;

  for (let i = 0; i < count; i++) {
    const baseAngle = (i / count) * Math.PI * 2 + time * speed;
    const x = centerX + orbitRadius * Math.cos(baseAngle);
    const y = centerY + orbitRadius * Math.sin(baseAngle);
    const alpha = isActive ? 0.35 + 0.15 * Math.sin(time * 1.2 + i) : 0.25;
    ctx.save();
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    ctx.arc(x, y, particleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

export function drawPlayIcon(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  size: number,
  color: string
): void {
  const [r, g, b] = hexToRgb(color);
  ctx.save();
  ctx.fillStyle = `rgba(${r},${g},${b},0.8)`;
  ctx.beginPath();
  const s = size * 0.5;
  ctx.moveTo(centerX - s * 0.6, centerY - s);
  ctx.lineTo(centerX - s * 0.6, centerY + s);
  ctx.lineTo(centerX + s * 0.8, centerY);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export function drawInterruptionRipple(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  progress: number,
  color: string
): void {
  if (progress >= 1) return;
  const [r, g, b] = hexToRgb(color);
  const ease = 1 - Math.pow(1 - progress, 2);
  const r2 = radius * (1 + ease * 0.25);
  ctx.save();
  ctx.strokeStyle = `rgba(${r},${g},${b},${0.6 * (1 - progress)})`;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(centerX, centerY, r2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}
