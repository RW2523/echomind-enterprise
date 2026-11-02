/** Per-viewer display preferences (theme, transparency). Browser storage is a convenience only:
 *  every read/write is guarded and the app renders correctly without it. */
export type ThemeMode = 'system' | 'light' | 'dark';
/** The colours: walnut, gold and black (the default since 2026-11-02, the owner's choice to try) or the earlier navy. */
export type Palette = 'walnut' | 'classic';
const THEME_KEY = 'echomind.law.theme';
const PALETTE_KEY = 'echomind.law.palette';
const TRANSPARENCY_KEY = 'echomind.law.transparency';

function read(key: string): string | null {
  try { return window.localStorage.getItem(key); } catch { return null; }
}
function write(key: string, value: string): void {
  try { window.localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

export function getThemeMode(): ThemeMode {
  const v = read(THEME_KEY);
  return v === 'light' || v === 'dark' ? v : 'system';
}
export function getPalette(): Palette {
  return read(PALETTE_KEY) === 'classic' ? 'classic' : 'walnut';
}
export function getReducedTransparency(): boolean {
  return read(TRANSPARENCY_KEY) === 'reduced';
}

export function applyPreferences(mode: ThemeMode = getThemeMode(), reduced = getReducedTransparency()): void {
  const root = document.documentElement;
  const dark = mode === 'dark' || (mode === 'system' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  root.dataset.theme = dark ? 'dark' : 'light';
  root.dataset.palette = getPalette();
  if (reduced) root.dataset.transparency = 'reduced';
  else delete root.dataset.transparency;
}

export function setThemeMode(mode: ThemeMode): void { write(THEME_KEY, mode); applyPreferences(mode); }
export function setPalette(palette: Palette): void { write(PALETTE_KEY, palette); applyPreferences(); }
export function setReducedTransparency(reduced: boolean): void {
  write(TRANSPARENCY_KEY, reduced ? 'reduced' : 'normal');
  applyPreferences(undefined, reduced);
}
