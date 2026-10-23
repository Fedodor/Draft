import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { OverlayWindow } from './components/OverlayWindow';
import { getCurrentWindow } from '@tauri-apps/api/window';
import './index.css';

// Branch on which window this webview is running in:
// the "overlay" window renders the compact live-draft overlay.
const isOverlay = getCurrentWindow().label === 'overlay';
if (isOverlay) {
  document.body.classList.add('overlay-mode');
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {isOverlay ? <OverlayWindow /> : <App />}
  </React.StrictMode>
);
