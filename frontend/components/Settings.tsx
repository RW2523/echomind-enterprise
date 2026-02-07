import React, { useState, useEffect, useCallback } from 'react';
import { AppSettings, PersonaType, PIPER_VOICES } from '../types';
import { getInstalledVoices, downloadVoice } from '../services/backend';

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
    <div className="h-full min-h-0 bg-[#0a0c1a]/20 overflow-y-auto">
      <div className="max-w-4xl mx-auto space-y-10 sm:space-y-12 py-2 pb-16">
        <section>
          <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Persona Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4">
            {personas.map((p) => (
              <button
                key={p}
                onClick={() => update('persona', p)}
                className={`p-6 rounded-3xl border transition-all text-left group ${
                  settings.persona === p
                    ? 'bg-cyan-500/10 border-cyan-500/40'
                    : 'bg-white/5 border-white/5 hover:border-white/10'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-bold ${settings.persona === p ? 'text-cyan-400' : 'text-slate-300'}`}>
                    {p}
                  </span>
                  {settings.persona === p && <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.8)]"></div>}
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Tailors reasoning style, vocabulary, and response tone for specialized workflows. Used in Knowledge Chat and Voice.
                </p>
              </button>
            ))}
          </div>
        </section>

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

        <section className="pt-6 sm:pt-8 border-t border-white/5">
          <div className="flex items-center justify-between p-4 sm:p-6 rounded-2xl sm:rounded-3xl bg-red-500/5 border border-red-500/10">
            <div>
              <h4 className="text-sm font-bold text-red-400">Developer Mode</h4>
              <p className="text-xs text-slate-500 mt-1">Access raw model parameters and debug tools.</p>
            </div>
            <button 
              onClick={() => update('developerMode', !settings.developerMode)}
              className={`w-14 h-8 rounded-full relative transition-colors ${settings.developerMode ? 'bg-red-500' : 'bg-slate-800'}`}
            >
              <div className={`absolute top-1 w-6 h-6 rounded-full bg-white shadow-lg transition-all ${settings.developerMode ? 'left-7' : 'left-1'}`}></div>
            </button>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Settings;
