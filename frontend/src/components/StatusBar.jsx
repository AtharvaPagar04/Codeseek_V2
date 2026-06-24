import { useState, useEffect } from 'react';
import { useHealth } from '../hooks/useHealth';
import { listProviderCredentials } from '../utils/api';

export default function StatusBar({
  ghUser,
  ghAvatarUrl,
  onConnectGitHub,
  onDisconnectGitHub,
  onToggleSidebar,
  isMobile,
  onOpenApiTokens,
  activeSession,
  githubNotice,
}) {
  const { status } = useHealth();
  const [providerMode, setProviderMode] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchMode = async () => {
      try {
        const creds = await listProviderCredentials();
        const active = creds.find(c => c.isActive);
        if (!cancelled && active) {
          setProviderMode(active.provider === 'local' ? 'Local' : 'Cloud');
        } else if (!cancelled) {
          setProviderMode(null);
        }
      } catch (e) {
        // ignore
      }
    };
    fetchMode();
    window.addEventListener('CODESEEK_PROVIDER_CHANGED', fetchMode);
    return () => {
      cancelled = true;
      window.removeEventListener('CODESEEK_PROVIDER_CHANGED', fetchMode);
    };
  }, []);

  return (
    <header className="flex flex-col shrink-0 z-20 border-b border-border bg-surface/80 backdrop-blur-md">
      <div className="flex items-center justify-between h-12 px-5">
      {/* Left: hamburger + app name + repo context */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onToggleSidebar}
          className="text-text-secondary hover:text-text-primary transition-colors p-1 -ml-1 shrink-0"
          aria-label="Toggle sidebar"
        >
          <HamburgerIcon />
        </button>
        <span className="font-mono text-sm font-semibold tracking-[0.15em] text-text-primary uppercase shrink-0">
          Codeseek
        </span>
        {activeSession && (
          <>
            <span className="text-text-muted text-xs select-none">/</span>
            <div className="min-w-0 flex flex-col leading-tight">
              <span className="font-mono text-sm text-text-primary truncate">{activeSession.repo_id}</span>
              <span className="text-2xs text-text-muted font-mono truncate hidden sm:block">{activeSession.repo_full_name}</span>
            </div>
            {activeSession.status && activeSession.status !== 'ready' && (
              <span className="text-2xs text-warning font-mono shrink-0 hidden sm:inline">
                {activeSession.status === 'failed' ? '● failed' : `● ${activeSession.status}`}
              </span>
            )}
          </>
        )}
        {!activeSession && (
          <span className="text-text-muted text-xs font-mono hidden sm:inline">v0.1</span>
        )}
      </div>

      {/* Right: health + GitHub */}
      <div className="flex items-center gap-4">
        {/* Health indicator */}
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${
              status === 'online'
                ? 'bg-online'
                : status === 'offline'
                ? 'bg-offline'
                : 'bg-text-muted animate-pulse'
            }`}
          />
          <div className="flex flex-col">
            <span className="text-xs text-text-muted hidden sm:inline leading-tight">
              {status === 'online' ? 'API Online' : status === 'offline' ? 'API Unreachable' : 'Checking…'}
            </span>
            {providerMode && status === 'online' && (
              <span className="text-2xs text-text-muted/70 font-mono hidden sm:inline leading-tight uppercase tracking-wider">
                {providerMode}
              </span>
            )}
          </div>
        </div>

        <div className="w-px h-4 bg-border" />

        {/* API Token config */}
        <button
          onClick={onOpenApiTokens}
          title="Manage API Tokens"
          className="text-xs text-text-secondary hover:text-text-primary transition-colors flex items-center gap-1.5"
        >
          <KeyIcon />
          <span className="hidden sm:inline">API Config</span>
        </button>

        <div className="w-px h-4 bg-border" />

        {/* GitHub status */}
        {ghUser ? (
          <div className="flex items-center gap-2">
            {ghAvatarUrl ? (
              <img
                src={ghAvatarUrl}
                alt={ghUser}
                className="w-6 h-6 rounded-full border border-border"
              />
            ) : (
              <div className="w-6 h-6 rounded-full bg-surface-3 border border-border flex items-center justify-center text-2xs text-text-muted font-mono">
                {ghUser[0]?.toUpperCase()}
              </div>
            )}
            <span className="text-xs text-text-secondary font-mono hidden sm:inline">{ghUser}</span>
            <button
              onClick={onDisconnectGitHub}
              className="text-xs text-text-muted hover:text-offline transition-colors font-mono"
              title="Disconnect GitHub"
            >
              Logout
            </button>
          </div>
        ) : (
          <button
            onClick={onConnectGitHub}
            className="text-xs text-text-primary bg-surface-3 border border-border rounded-full px-3 py-1 hover:bg-surface-2 hover:border-text-muted transition-colors"
          >
            Connect GitHub
          </button>
        )}
      </div>
      </div>
      {githubNotice && (
        <div className="border-t border-border/60 px-5 py-2 text-[11px] font-mono text-warning">
          {githubNotice}
        </div>
      )}
    </header>
  );
}

function HamburgerIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor" aria-hidden="true">
      <rect y="3" width="18" height="1.5" rx="0.75" />
      <rect y="8.25" width="18" height="1.5" rx="0.75" />
      <rect y="13.5" width="18" height="1.5" rx="0.75" />
    </svg>
  );
}

function KeyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3.5 11.5a3.5 3.5 0 1 1 3.163-5H14L15.5 8 14 9.5l-1-1-1 1-1-1-1 1-1-1-1.5 1.5h-.837A3.5 3.5 0 0 1 3.5 11.5zM6 8a2 2 0 1 0-4 0 2 2 0 0 0 4 0z" />
    </svg>
  );
}
