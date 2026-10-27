/**
 * The line drawing behind the home: a courthouse front (pediment, six fluted columns, steps), drawn here so nothing is
 * fetched. Decorative only: hidden from assistive technology and painted in the current colour at low opacity by CSS.
 */
import React from 'react';

const COLUMNS = [150, 218, 286, 354, 422, 490];

export default function Facade({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 640 360" aria-hidden="true" focusable="false">
      <g fill="none" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round">
        <path d="M112 118 L320 34 L528 118 Z" />
        <path d="M150 112 L320 46 L490 112" opacity=".55" />
        <circle cx="320" cy="88" r="11" opacity=".7" />
        <path d="M312 88 h16 M320 80 v16" opacity=".5" />
        <rect x="104" y="118" width="432" height="14" />
        <path d="M104 138 H536" />
        <path d="M110 132 v6 M530 132 v6" opacity=".6" />
        {COLUMNS.map((x) => (
          <g key={x}>
            <path d={`M${x - 16} 144 h32 M${x - 13} 150 h26`} />
            <path d={`M${x - 11} 150 V298 M${x + 11} 150 V298`} />
            <path d={`M${x - 4} 156 V292 M${x + 4} 156 V292`} opacity=".45" />
            <path d={`M${x - 13} 298 h26 M${x - 16} 304 h32`} />
          </g>
        ))}
        <path d="M96 312 H544 M84 322 H556 M72 332 H568" />
        <path d="M60 342 H580" opacity=".5" />
      </g>
    </svg>
  );
}
