/** The AJACE mark: a pediment's "A" over a column base, drawn as lines so it stays sharp at any size (2026-11-05). */
export default function BrandMark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden focusable="false">
      <path d="M4.5 27.5 16 4.5l11.5 23" stroke="currentColor" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />
      <path d="M10.2 19.5h11.6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M2.5 27.5h27" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity=".55" />
    </svg>
  );
}
