import React from 'react';
import { ArrowUpRight, Box, Compass, Sparkles } from 'lucide-react';

export default function Navbar({
  health,
  onReturnToShowcase,
  pageView = 'floorplan',
  setPageView,
}) {
  const online = health?.status === 'healthy';
  return (
    <header className="studio-nav">
      <a
        className="studio-brand"
        href="#showcase"
        onClick={(event) => {
          event.preventDefault();
          onReturnToShowcase?.();
        }}
        aria-label="ArchiGen home"
      >
        <span className="brand-symbol">
          <Compass size={22} strokeWidth={1.5} />
        </span>
        <span>
          ARCHI<span className="brand-accent">GEN</span>
          <small>PLAN · 3D · FACADE STUDIO</small>
        </span>
      </a>
      <nav className="workspace-navigation" aria-label="Main workspace">
        <button
          data-testid="nav-floorplan"
          aria-current={pageView === 'floorplan' ? 'page' : undefined}
          onClick={() => setPageView('floorplan')}
        >
          <Box size={16} />
          Floorplan & 3D
        </button>
        <button
          data-testid="nav-generative"
          aria-current={pageView === 'generative_studio' ? 'page' : undefined}
          onClick={() => setPageView('generative_studio')}
        >
          <Sparkles size={16} />
          AI Generative Studio
        </button>
      </nav>
      <div className="nav-utilities">
        <span className={`connection-status ${online ? 'is-online' : ''}`} role="status">
          <i />
          {online ? 'Connected' : 'Connecting'}
        </span>
        <button className="showcase-link" onClick={onReturnToShowcase}>
          Home <ArrowUpRight size={15} />
        </button>
      </div>
    </header>
  );
}
