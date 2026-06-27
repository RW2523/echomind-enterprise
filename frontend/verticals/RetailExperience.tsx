import React from 'react';
import { resolvePack } from '../packs';
import { Ambient, Nav, Hero, FeatureGrid, AnalyzerWidget, VerticalChat, Footer, useChatId } from './_shared';

const FEATURES = [
  { icon: '🔎', title: 'Product finder', body: 'Matches customer needs to the right products or vehicles from your catalog.' },
  { icon: '🧾', title: 'Instant quotes', body: 'Specs, pricing, warranty, and financing assembled into a clean quote.' },
  { icon: '🚗', title: 'Live inventory', body: 'Answers from your current inventory and trims — no guesswork on the floor.' },
  { icon: '🔒', title: 'On-prem & branded', body: 'Runs on an in-store box; a private, branded concierge at counter or kiosk.' },
];
const STATS = [{ k: '100%', v: 'on-premise' }, { k: '24/7', v: 'sales concierge' }, { k: 'instant', v: 'quotes' }];
const SAMPLE = `Customer is looking for a mid-range smart thermostat for a 2-zone home, and an EV under $45,000 with monthly financing around $600.`;

const RetailExperience: React.FC = () => {
  const pack = resolvePack();
  const accent = pack?.accent ?? '#f59e0b';
  const chatId = useChatId('Sales concierge');
  return (
    <div className="h-full w-full overflow-y-auto bg-[#05070a] text-slate-200" style={{ height: '100dvh' }}>
      <Ambient />
      <div className="relative z-10">
        <Nav name={pack?.name ?? 'EchoMind Retail'} accent={accent}
          links={[{ label: 'Product finder', id: 'analyzer' }, { label: 'Ask the concierge', id: 'copilot', primary: true }]} />
        <Hero accent={accent} eyebrow="AI sales concierge"
          titleA="Your AI sales concierge, on the " titleHi="floor" titleB="."
          sub="Match customers to the right products, build quotes with financing, and answer inventory questions instantly — from your catalog, on a private in-store box."
          ctas={[{ label: 'Find a product', id: 'analyzer', primary: true }, { label: 'Ask the concierge', id: 'copilot' }]} stats={STATS} />
        <FeatureGrid features={FEATURES} />
        <AnalyzerWidget id="analyzer" accent={accent} chatId={chatId}
          heading="Product & vehicle finder" sub="Describe what the customer wants; get matched products with specs, price, and financing."
          label="Customer needs" placeholder="Describe the customer's needs…" sample={SAMPLE} buttonLabel="Find matches & quote"
          buildPrompt={(t) => `A customer is looking for: ${t}. From our catalog and inventory, recommend matching products/vehicles with a table (Item | Key specs | Price | Warranty), then outline applicable financing options. Use only catalog facts.`} />
        <VerticalChat id="copilot" accent={accent} chatId={chatId} persona={pack?.persona}
          heading="Ask the sales concierge" sub="Grounded in your catalog & inventory."
          starters={['What products do we carry?', 'Compare the financing options', 'What vehicles are in inventory right now?']}
          placeholder="Ask about products, inventory, financing…" />
        <Footer name={pack?.name ?? 'EchoMind Retail'} />
      </div>
    </div>
  );
};
export default RetailExperience;
