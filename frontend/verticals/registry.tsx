import React from 'react';
import BankExperience from './BankExperience';

/**
 * Per-vertical bespoke experiences. When a pack is active AND it has an entry here, App renders this
 * full-screen experience instead of the standard app. Verticals without an entry fall back to the
 * standard themed app, so we can ship one bespoke vertical at a time without breaking the others.
 */
export const VERTICAL_EXPERIENCES: Record<string, React.FC> = {
  bank: BankExperience,
};
