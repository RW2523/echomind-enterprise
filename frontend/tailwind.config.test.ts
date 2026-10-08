import { describe, expect, it } from 'vitest';
import config from './tailwind.config.js';

describe('tailwind content globs', () => {
  it('scan the vertical pages and the Law workspace (audit R-12)', () => {
    const globs = (config as { content: string[] }).content;
    expect(globs).toContain('./verticals/**/*.{js,ts,jsx,tsx}');
    expect(globs).toContain('./law/**/*.{js,ts,jsx,tsx}');
  });
});
