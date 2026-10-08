import './index.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { installSessionFetch } from './services/session';

// CSRF header on same-origin writes + session-expiry signal, before any request is made.
installSessionFetch();

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
