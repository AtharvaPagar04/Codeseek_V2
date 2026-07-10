import React from 'react';
import { useState, useEffect, useRef } from 'react';

export default function RepoPickerModal({
  isConnected,
  repos,
  reposLoading,
  reposError,
  onSelect,
  onClose,
  onConnectGitHub,
  onLoadRepos,
  oauthLoading = false,
  oauthError = null,
}) {
  const [filter, setFilter] = useState('');
  const inputRef = useRef(null);
  const overlayRef = useRef(null);

  // Fetch repos on open if connected
  useEffect(() => {
    if (!isConnected) return;
    onLoadRepos?.();
  }, [isConnected, onLoadRepos]);

  // Focus search input and handle Escape
  useEffect(() => {
    inputRef.current?.focus();
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const filtered = repos.filter((r) =>
    r.name.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-40 bg-black/60 flex items-start justify-center pt-[10vh]"
      onClick={(e) => e.target === overlayRef.current && onClose()}
    >
      <div className="bg-surface-2 border border-border rounded-2xl w-full max-w-lg mx-4 shadow-xl animate-fadeIn flex flex-col max-h-[75vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <span className="text-sm font-medium text-text-primary">New Session</span>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors text-lg leading-none">
            ×
          </button>
        </div>

        {!isConnected ? (
          // Not connected state
          <div className="flex flex-col items-center py-8 px-6 gap-3">
            <p className="text-text-secondary text-sm text-center">
              Connect your GitHub account to create sessions.
            </p>
            <button
              onClick={onConnectGitHub}
              disabled={oauthLoading}
              className="flex items-center gap-2 px-4 py-2 text-sm text-text-primary bg-surface-3 border border-border rounded-xl hover:bg-surface-2 hover:border-text-muted transition-colors font-semibold disabled:opacity-60 disabled:cursor-wait"
            >
              {oauthLoading ? (
                <>
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-text-muted border-t-text-primary animate-spin" />
                  Connecting…
                </>
              ) : (
                <>Connect via GitHub</>
              )}
            </button>
            {oauthError && (
              <p className="text-xs text-offline/90 font-mono text-center">⚠ {oauthError}</p>
            )}
          </div>
        ) : (
          <>
            {/* Search */}
            <div className="px-3 py-2 border-b border-border shrink-0">
              <input
                ref={inputRef}
                type="text"
                placeholder="Filter repositories…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="w-full bg-surface-3 border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted transition-colors"
              />
            </div>

            {/* Body */}
            <div className="overflow-y-auto flex-1">
              {reposLoading && (
                <div className="flex flex-col gap-2 p-3">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="h-12 bg-surface-3 rounded-xl animate-pulse" />
                  ))}
                </div>
              )}

              {reposError && (
                <div className="p-4 text-sm text-offline/80 text-center space-y-3">
                  <div>{reposError}</div>
                  <button
                    onClick={() => onLoadRepos?.()}
                    className="rounded-full border border-border px-3 py-1 text-xs font-mono text-text-secondary hover:text-text-primary hover:border-text-muted transition-colors"
                  >
                    Retry Repo Load
                  </button>
                </div>
              )}

              {!reposLoading && !reposError && filtered.length === 0 && (
                <div className="p-4 text-sm text-text-muted text-center">
                  {filter ? `No repos matching "${filter}"` : 'No repositories found.'}
                </div>
              )}

              {!reposLoading && !reposError && filtered.map((repo) => (
                <button
                  key={repo.id}
                  onClick={() => onSelect(repo)}
                  className="w-full text-left px-4 py-3 hover:bg-surface-3 transition-colors border-b border-border/40 last:border-0"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-sm text-text-primary font-medium truncate">
                      {repo.name}
                    </span>
                    <span
                      className={`shrink-0 text-2xs px-2 py-0.5 rounded-full border ${
                        repo.private
                          ? 'text-warning border-warning/30 bg-warning/5'
                          : 'text-text-muted border-border'
                      }`}
                    >
                      {repo.private ? 'Private' : 'Public'}
                    </span>
                  </div>
                  {repo.description && (
                    <div className="text-xs text-text-muted mt-0.5 truncate">{repo.description}</div>
                  )}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
