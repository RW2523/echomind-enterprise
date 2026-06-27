import React from 'react';
import { resolvePack } from '../packs';
import { Ambient, Nav, Hero, FeatureGrid, AnalyzerWidget, VerticalChat, Footer, useChatId } from './_shared';

const FEATURES = [
  { icon: '🎙️', title: 'Capture the room', body: 'Live transcription of the meeting — diarized, private, fully on-prem.' },
  { icon: '✅', title: 'Action items', body: 'Owners and due dates extracted automatically, ready to distribute.' },
  { icon: '🧭', title: 'Decisions tracked', body: 'Every decision captured and searchable against your company knowledge.' },
  { icon: '🔒', title: 'Never leaves the room', body: 'Runs on a box in the building — no meeting audio or notes hit the cloud.' },
];
const STATS = [{ k: '100%', v: 'on-premise' }, { k: '0', v: 'data to cloud' }, { k: 'every', v: 'meeting actioned' }];
const SAMPLE = `Q3 Product Strategy — attendees: Linnea (Eng), Mikael (Security), Dana (PM).
Decision: ship the offline sync engine in Q4. Decision: pursue SOC 2 Type II.
Linnea to deliver the sync engine to QA by Aug 22. Mikael to complete the SOC 2 evidence package by Sep 5.`;

const MeetingsExperience: React.FC = () => {
  const pack = resolvePack();
  const accent = pack?.accent ?? '#06b6d4';
  const chatId = useChatId('Meeting copilot');
  return (
    <div className="h-full w-full overflow-y-auto bg-[#05070a] text-slate-200" style={{ height: '100dvh' }}>
      <Ambient />
      <div className="relative z-10">
        <Nav name={pack?.name ?? 'EchoMind Meeting Rooms'} accent={accent}
          links={[{ label: 'Action items', id: 'analyzer' }, { label: 'Ask the copilot', id: 'copilot', primary: true }]} />
        <Hero accent={accent} eyebrow="Private boardroom intelligence"
          titleA="Every meeting, captured and " titleHi="actioned" titleB="."
          sub="Capture the room, surface decisions and an owner / due-date action list, and brief from your company knowledge — all on-premise, never leaving the building."
          ctas={[{ label: 'Extract action items', id: 'analyzer', primary: true }, { label: 'Ask the copilot', id: 'copilot' }]} stats={STATS} />
        <FeatureGrid features={FEATURES} />
        <AnalyzerWidget id="analyzer" accent={accent} chatId={chatId}
          heading="Action items in one click" sub="Paste meeting notes; get decisions and an owner / due-date action list."
          label="Meeting notes" placeholder="Paste meeting notes…" sample={SAMPLE} buttonLabel="Extract decisions & actions"
          buildPrompt={(t) => `From these meeting notes, produce: (1) a short 'Decisions' list, and (2) an 'Action Items' table with columns Action | Owner | Due date | Status. Use only what is in the notes.\n\nNotes:\n${t}`} />
        <VerticalChat id="copilot" accent={accent} chatId={chatId} persona={pack?.persona}
          heading="Ask the meeting copilot" sub="Grounded in your minutes & company policies."
          starters={['Summarize the Q3 product strategy meeting', 'What action items were assigned, and to whom?', 'What is our remote-work policy?']}
          placeholder="Ask about decisions, actions, policies…" />
        <Footer name={pack?.name ?? 'EchoMind Meeting Rooms'} />
      </div>
    </div>
  );
};
export default MeetingsExperience;
