// @types/react-dom is not installed in this project, so the main (non-strict) build already treats 'react-dom' as `any`.
// This says so explicitly, which lets the strict build of law/counsel (tsconfig.counsel.json) type-check the existing
// Law files it imports. It changes nothing for the main build. Replace it with @types/react-dom when that is approved.
declare module 'react-dom';
