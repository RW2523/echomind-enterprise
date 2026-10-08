/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./hooks/**/*.{js,ts,jsx,tsx}",
    "./services/**/*.{js,ts,jsx,tsx}",
    "./utils/**/*.{js,ts,jsx,tsx}",
    // Vertical pages and the Law workspace: without these globs their classes were never generated
    // and the bespoke vertical pages rendered partly unstyled in production builds (audit R-12).
    "./verticals/**/*.{js,ts,jsx,tsx}",
    "./law/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Per-vertical accent, driven by CSS vars set from the active pack (see packs.ts / index.css).
        accent: 'rgb(var(--accent-rgb) / <alpha-value>)',
        accent2: 'rgb(var(--accent2-rgb) / <alpha-value>)',
      },
    },
  },
  plugins: [],
};
