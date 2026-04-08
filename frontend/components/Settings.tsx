import React, { useState, useEffect, useCallback } from 'react';
import { AppSettings, PersonaType, PIPER_VOICES } from '../types';
import { getInstalledVoices, downloadVoice, addSampleTranscripts } from '../services/backend';

interface PersonaMeta {
  icon: string;
  title: string;
  description: string;
  accentColor: string;
  borderColor: string;
  activeBg: string;
  activeBorder: string;
}

const PERSONA_META: Record<PersonaType, PersonaMeta> = {
  [PersonaType.TEACHER]: {
    icon: '🎓',
    title: 'Teacher / Professor',
    description: 'Explains any topic with clarity and depth. Uses analogies, step-by-step breakdowns, and an encouraging academic tone. Perfect for learning and understanding complex concepts.',
    accentColor: 'text-emerald-400',
    borderColor: 'border-emerald-500/40',
    activeBg: 'bg-emerald-500/10',
    activeBorder: 'border-emerald-500/40',
  },
  [PersonaType.FINANCIAL]: {
    icon: '💼',
    title: 'Financial Advisor',
    description: 'Expert in DoD FMR, government financial regulations, compliance, and regulatory matters. Cites sections precisely and provides authoritative financial guidance.',
    accentColor: 'text-cyan-400',
    borderColor: 'border-cyan-500/40',
    activeBg: 'bg-cyan-500/10',
    activeBorder: 'border-cyan-500/40',
  },
  [PersonaType.FUNNY]: {
    icon: '😄',
    title: 'Funny & Calming Assistant',
    description: 'Warm, witty, and genuinely helpful. Blends tasteful humor with real assistance to keep things light and stress-free. Great for everyday questions and when you need a calming presence.',
    accentColor: 'text-amber-400',
    borderColor: 'border-amber-500/40',
    activeBg: 'bg-amber-500/10',
    activeBorder: 'border-amber-500/40',
  },
  [PersonaType.LAWYER]: {
    icon: '⚖️',
    title: 'Lawyer',
    description: 'Experienced legal advisor specializing in contracts, compliance, regulations, and risk analysis. Uses IRAC structure, cites statutes precisely, and always includes appropriate legal disclaimers.',
    accentColor: 'text-violet-400',
    borderColor: 'border-violet-500/40',
    activeBg: 'bg-violet-500/10',
    activeBorder: 'border-violet-500/40',
  },
  [PersonaType.AI_EXPERT]: {
    icon: '🤖',
    title: 'AI Expert & Manager',
    description: 'Senior AI/ML engineer and software manager. Advises on AI architectures, system design, engineering best practices, technical roadmaps, and team leadership with hands-on depth.',
    accentColor: 'text-rose-400',
    borderColor: 'border-rose-500/40',
    activeBg: 'bg-rose-500/10',
    activeBorder: 'border-rose-500/40',
  },
};

interface SettingsProps {
  settings: AppSettings;
  setSettings: (s: AppSettings) => void;
}

const Settings: React.FC<SettingsProps> = ({ settings, setSettings }) => {
  const contextWindows: AppSettings['contextWindow'][] = ['24h', '48h', '1w', 'all'];
  const personas = Object.values(PersonaType);
  const [installedVoiceIds, setInstalledVoiceIds] = useState<Set<string>>(new Set());
  const [downloadingVoiceId, setDownloadingVoiceId] = useState<string | null>(null);
  const [voicesLoadError, setVoicesLoadError] = useState<string | null>(null);
  const [addingSamples, setAddingSamples] = useState(false);

  const loadInstalledVoices = useCallback(async () => {
    try {
      setVoicesLoadError(null);
      const { voice_ids } = await getInstalledVoices();
      setInstalledVoiceIds(new Set(voice_ids || []));
    } catch (e) {
      setVoicesLoadError((e as Error)?.message || 'Could not load voice list');
      setInstalledVoiceIds(new Set());
    }
  }, []);

  useEffect(() => {
    loadInstalledVoices();
  }, [loadInstalledVoices]);

  const update = (key: keyof AppSettings, val: AppSettings[keyof AppSettings]) => {
    setSettings({ ...settings, [key]: val });
  };

  const selectVoice = async (voiceId: string) => {
    if (downloadingVoiceId) return;
    const installed = installedVoiceIds.has(voiceId);
    if (installed) {
      update('voiceName', voiceId);
      return;
    }
    setDownloadingVoiceId(voiceId);
    try {
      await downloadVoice(voiceId);
      setInstalledVoiceIds((prev) => new Set([...prev, voiceId]));
      update('voiceName', voiceId);
    } catch (e) {
      alert((e as Error)?.message || 'Voice download failed');
    } finally {
      setDownloadingVoiceId(null);
    }
  };

  return (
    <div className="h-full min-h-0 bg-[#0a0c1a]/20 overflow-y-auto overflow-x-hidden">
      <div className="max-w-4xl mx-auto space-y-10 sm:space-y-12 py-2 px-1 sm:px-0 pb-16">
        <section>
          <h3 className="text-lg sm:text-xl font-bold text-white mb-1 sm:mb-2">Persona Configuration</h3>
          <p className="text-xs text-slate-500 mb-4 sm:mb-6">Choose the AI persona for Knowledge Chat and Voice. Each persona has its own expertise, tone, and guardrails.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4">
            {personas.map((p) => {
              const meta = PERSONA_META[p as PersonaType];
              const isActive = settings.persona === p;
              return (
                <button
                  key={p}
                  type="button"
                  onClick={() => update('persona', p)}
                  className={`p-4 sm:p-5 rounded-2xl sm:rounded-3xl border transition-all text-left group touch-manipulation ${
                    isActive
                      ? `${meta.activeBg} ${meta.activeBorder}`
                      : 'bg-white/5 border-white/5 hover:border-white/10 hover:bg-white/8'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2 gap-2">
                    <div className="flex items-center gap-2.5">
                      <span className="text-xl leading-none">{meta.icon}</span>
                      <span className={`text-sm font-bold ${isActive ? meta.accentColor : 'text-slate-300'}`}>
                        {meta.title}
                      </span>
                    </div>
                    {isActive && (
                      <div className={`w-2 h-2 mt-1 shrink-0 rounded-full ${meta.accentColor.replace('text-', 'bg-')} shadow-lg`} />
                    )}
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed pl-8">
                    {meta.description}
                  </p>
                </button>
              );
            })}
          </div>
        </section>

        {/* <section>
          <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Voice Context / Role (System Prompt)</h3>
          <div className="glass rounded-2xl sm:rounded-3xl p-5 sm:p-6 md:p-8 space-y-4">
            <p className="text-xs text-slate-500 -mt-2">Set the assistant&apos;s role and context for Voice AI Conversation. This is saved and retained until you clear it.</p>
            <textarea
              value={settings.voiceContext ?? ''}
              onChange={(e) => update('voiceContext', e.target.value)}
              placeholder="Example: You are a car dealership sales agent. Ask 1-2 questions, then recommend a car."
              className="w-full h-28 min-h-[6rem] rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 resize-y outline-none focus:border-cyan-500/50"
            />
            <button
              type="button"
              onClick={() => update('voiceContext', '')}
              className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-medium bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors touch-manipulation"
            >
              Clear context
            </button>
          </div>
        </section>

        <section>
          <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Voice Assistant Identity</h3>
          <div className="glass rounded-2xl sm:rounded-3xl p-5 sm:p-6 md:p-8 space-y-4 mb-8">
            <div>
              <label className="text-sm font-bold text-slate-300">Assistant name</label>
              <p className="text-xs text-slate-500 mt-0.5">Give your assistant a name (e.g. Alex). Talk naturally—say the name or just speak; say &quot;stop&quot; to pause, &quot;start&quot; or the name to continue.</p>
              <input
                type="text"
                value={settings.voiceBotName ?? ''}
                onChange={(e) => update('voiceBotName', e.target.value)}
                placeholder="e.g. Alex"
                className="mt-2 w-full max-w-xs rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 outline-none focus:border-cyan-500/50"
              />
            </div>
            <div>
              <label className="text-sm font-bold text-slate-300">Your name (optional)</label>
              <p className="text-xs text-slate-500 mt-0.5">The assistant can remember and use your name in conversation.</p>
              <input
                type="text"
                value={settings.voiceUserName ?? ''}
                onChange={(e) => update('voiceUserName', e.target.value)}
                placeholder="Your name"
                className="mt-2 w-full max-w-xs rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 outline-none focus:border-cyan-500/50"
              />
            </div>
          </div>
        </section> */}

        <section>
          <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Voice & Audio (Piper TTS)</h3>
          <div className="glass rounded-2xl sm:rounded-3xl p-5 sm:p-6 md:p-8 space-y-6">
            <div className="flex flex-col gap-4">
              <label className="text-sm font-bold text-slate-300">Piper Voice (en_US)</label>
              <p className="text-xs text-slate-500 -mt-2">Select the TTS voice for Voice Conversation. The chosen voice is downloaded automatically when selected.</p>
              {voicesLoadError && (
                <p className="text-xs text-amber-400">Voice server: {voicesLoadError}. You can still select a voice; it will be downloaded when the server is available.</p>
              )}
              {downloadingVoiceId && (
                <p className="text-xs text-cyan-400">Downloading voice…</p>
              )}
              <div className="flex flex-wrap gap-3 max-h-48 overflow-y-auto">
                {PIPER_VOICES.map((v) => {
                  const isInstalled = installedVoiceIds.has(v.id);
                  const isDownloading = downloadingVoiceId === v.id;
                  return (
                    <button
                      key={v.id}
                      onClick={() => selectVoice(v.id)}
                      disabled={isDownloading}
                      className={`px-4 py-2 rounded-2xl border text-sm font-semibold transition-all shrink-0 ${
                        settings.voiceName === v.id
                          ? 'bg-violet-600 border-violet-500 text-white shadow-lg'
                          : isInstalled
                            ? 'bg-white/5 border-white/5 text-slate-400 hover:text-white hover:bg-white/10'
                            : 'bg-white/5 border-amber-500/30 text-slate-500 hover:text-amber-400 hover:border-amber-500/50'
                      }`}
                    >
                      {v.label}
                      {isInstalled && settings.voiceName !== v.id && <span className="ml-1.5 text-[10px] text-slate-500">✓</span>}
                      {isDownloading && ' …'}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        <section>
          <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Knowledge Base Context</h3>
          <div className="glass rounded-2xl sm:rounded-3xl p-5 sm:p-6 md:p-8">
            <div className="flex flex-col gap-4">
              <label className="text-sm font-bold text-slate-300">Retrieval Window</label>
              <div className="flex bg-slate-900/50 p-1.5 rounded-2xl border border-white/5">
                {contextWindows.map((cw) => (
                  <button
                    key={cw}
                    onClick={() => update('contextWindow', cw)}
                    className={`flex-1 py-2 rounded-xl text-xs font-bold uppercase tracking-wider transition-all ${
                      settings.contextWindow === cw
                        ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/20'
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {cw === 'all' ? 'All Time' : cw}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-500 mt-2 italic px-2">
                Limits the RAG search to specific date ranges for more relevant temporal grounding.
              </p>
            </div>
          </div>
        </section>


        <section className="pt-4 sm:pt-6">
          <div className="glass rounded-2xl sm:rounded-3xl p-5 sm:p-6 md:p-8 border border-amber-500/10 bg-amber-500/5">
            <h4 className="text-sm font-bold text-amber-400 mb-2">RAG Testing</h4>
            <p className="text-xs text-slate-500 mb-4">
              Add sample transcript chunks to the embedding index. Chunks span the last 48 hours plus fixed dates (Dec 1, 2025; Oct 10, 2025). Test queries like &quot;last 2 hours&quot;, &quot;2 Dec 2025&quot;, or &quot;10 Oct 2025&quot;.
            </p>
            <button
              type="button"
              disabled={addingSamples}
              onClick={async () => {
                setAddingSamples(true);
                try {
                  const res = await addSampleTranscripts();
                  window.dispatchEvent(new CustomEvent('echomind:transcripts-added', { detail: res.added }));
                  alert(`Added ${res.added} sample transcripts. They appear in the Transcripts tab. Try queries like "last 2 hours" or "pricing" in chat.`);
                } catch (e) {
                  alert((e as Error)?.message || 'Failed to add sample transcripts');
                } finally {
                  setAddingSamples(false);
                }
              }}
              className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30 transition-colors touch-manipulation disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {addingSamples ? 'Adding…' : 'Add sample transcripts'}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Settings;
