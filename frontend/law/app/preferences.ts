/** Per-viewer display preferences (theme, transparency). Browser storage is a convenience only:
 *  every read/write is guarded and the app renders correctly without it. */
export type ThemeMode = 'system' | 'light' | 'dark';
const THEME_KEY = 'echomind.law.theme';
const TRANSPARENCY_KEY = 'echomind.law.transparency';

function read(key: string): string | null {
  try { return window.localStorage.getItem(key); } catch { return null; }
}
function write(key: string, value: string): void {
  try { window.localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

/** EchoMind Legal opens in its navy "Chambers" appearance until the person chooses another (2026-10-28, owner's request). */
export function getThemeMode(): ThemeMode {
  const v = read(THEME_KEY);
  return v === 'light' || v === 'dark' || v === 'system' ? v : 'dark';
}
export function getReducedTransparency(): boolean {
  return read(TRANSPARENCY_KEY) === 'reduced';
}

export function applyPreferences(mode: ThemeMode = getThemeMode(), reduced = getReducedTransparency()): void {
  const root = document.documentElement;
  const dark = mode === 'dark' || (mode === 'system' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  root.dataset.theme = dark ? 'dark' : 'light';
  if (reduced) root.dataset.transparency = 'reduced';
  else delete root.dataset.transparency;
}

export function setThemeMode(mode: ThemeMode): void { write(THEME_KEY, mode); applyPreferences(mode); }
export function setReducedTransparency(reduced: boolean): void {
  write(TRANSPARENCY_KEY, reduced ? 'reduced' : 'normal');
  applyPreferences(undefined, reduced);
}
