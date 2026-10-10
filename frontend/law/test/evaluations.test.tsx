import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, screen, within } from '@testing-library/react';
import axe from 'axe-core';
import type { EvaluationRun } from '../api/types';
import { me, mockApi, renderLaw } from './harness';

/** S2.8 — evaluation runs, for the firm's audit readers (D-044). */
const RUNS: EvaluationRun[] = [
  { id: 'r1', suite: 'research', set: 'heldout', commit: 'abc1234', models: { model: 'nvidia/Qwen3-30B-A3B-FP4', embeddings: 'nomic-embed-text' },
    dataset_sha256: 'd'.repeat(64), report: '20261010-120000-heldout.json', report_sha256: 'e'.repeat(64), run_at: '20261010-120000',
    imported_at: '2026-10-10T12:10:00.000Z', summary: { answer_accuracy: '28/29', citation_precision: 0.93, t06_leaks: 0, items: 32 } },
  { id: 'r2', suite: 'compare', set: 'golden', commit: 'abc1234', models: {}, dataset_sha256: null, report: 'x.json', report_sha256: 'f'.repeat(64),
    run_at: '20261010-121500', imported_at: '2026-10-10T12:20:00.000Z', summary: { alignment: { precision: 1.0, recall: 0.95 } } },
];

describe('Evaluations (S2.8)', () => {
  afterEach(() => cleanup());

  it('lists each run with its headline measures, set, commit and models', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['managing_partner'])], 'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/evaluations': () => [200, { runs: RUNS }] });
    const { container } = renderLaw('/admin/evaluations');
    const table = await screen.findByRole('table', { name: 'Evaluation runs' });
    const [, research, compare] = within(table).getAllByRole('row');
    expect(research.textContent).toMatch(/Research answers.*Held-out.*answer accuracy28\/29.*citation precision0\.93.*t06 leaks0/);
    expect(research.textContent).not.toContain('items');                      // headline measures only
    expect(research.textContent).toContain('abc1234');
    expect(compare.textContent).toMatch(/Compare and conflicts.*Tuning set.*alignment · precision1/);
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('is not available to a lawyer without a firm audit role', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/admin/evaluations');
    expect(await screen.findByText(/not available/i)).toBeTruthy();
  });
});
