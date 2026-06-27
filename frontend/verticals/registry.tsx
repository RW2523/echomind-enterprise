import React from 'react';
import BankExperience from './BankExperience';
import HealthExperience from './HealthExperience';
import MeetingsExperience from './MeetingsExperience';
import LawExperience from './LawExperience';
import RetailExperience from './RetailExperience';

/**
 * Per-vertical bespoke experiences. When a pack is active AND it has an entry here, App renders this
 * full-screen experience instead of the standard app. Verticals without an entry fall back to the
 * standard themed app.
 */
export const VERTICAL_EXPERIENCES: Record<string, React.FC> = {
  bank: BankExperience,
  health: HealthExperience,
  meetings: MeetingsExperience,
  law: LawExperience,
  retail: RetailExperience,
};
