import React from 'react';
import { useState } from 'react';
import ConfirmDialog from './ConfirmDialog';
import { formatDistanceToNow } from 'date-fns';

/**
 * Strip markdown syntax for the last-message excerpt preview.
 */
function stripMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/```[\s\S]*?```/g, '[code]')
    .replace(/`[^`]+`/g, (m) => m.slice(1, -1))
    .replace(/[*_~#>\-]/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

function getExcerpt(messages) {
  if (!messages || messages.length === 0) return 'No messages yet';
  const last = [...messages].reverse().find((m) => m.content && !m.loading);
  if (!last) return 'No messages yet';
  const stripped = stripMarkdown(last.content);
  return stripped.length > 60 ? stripped.slice(0, 60) + '…' : stripped;
}

function getSessionExcerpt(session) {
  const activeThread =
    session.threads?.find((thread) => thread.id === session.active_thread_id) ||
    session.threads?.[0] ||
    null;
  return getExcerpt(activeThread?.messages || []);
}

function statusBadge(session) {
  if (session.status === 'failed') {
    return { label: 'Failed', className: 'text-offline border-offline/30 bg-offline/10' };
  }
  if (session.status && session.status !== 'ready') {
    return { label: 'Indexing', className: 'text-warning border-warning/30 bg-warning/10' };
  }
  
  const freshness = session.repo_status?.status;
  if (freshness === 'up_to_date') {
    return { label: 'Fresh', className: 'text-online border-online/30 bg-online/10' };
  }
  if (freshness === 'out_of_date') {
    return { label: 'Stale', className: 'text-warning border-warning/30 bg-warning/10' };
  }
  if (freshness === 'dirty_worktree') {
    return { label: 'Dirty', className: 'text-offline border-offline/30 bg-offline/10' };
  }
  if (freshness === 'indexing') {
    return { label: 'Indexing', className: 'text-warning border-warning/30 bg-warning/10' };
  }
  if (freshness === 'failed') {
    return { label: 'Failed', className: 'text-offline border-offline/30 bg-offline/10' };
  }
  if (freshness === 'unknown' || freshness === 'error' || freshness === 'missing') {
    return { label: 'Unknown', className: 'text-text-muted border-border bg-surface-2' };
  }

  return { label: 'Ready', className: 'text-online border-online/30 bg-online/10' };
}

function getRelativeTime(iso) {
  if (!iso) return '';
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true });
  } catch {
    return '';
  }
}

export default function SessionItem({
  session,
  isActive,
  onSelect,
  onDelete,
}) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const badge = statusBadge(session);

  const isIndexing = session.status === 'indexing';

  const handleDelete = (e) => {
    e.stopPropagation();
    if (isIndexing) return;
    setConfirmOpen(true);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect();
    }
  };

  return (
    <>
      <div
        role="button"
        tabIndex={0}
        onClick={onSelect}
        onKeyDown={handleKeyDown}
        className={`group w-full text-left px-3 py-2.5 rounded-xl transition-all duration-150 relative ${
          isActive
            ? 'bg-surface-3 border border-border text-text-primary'
            : 'border border-transparent hover:bg-surface-2 text-text-secondary hover:text-text-primary'
        }`}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="font-medium text-sm truncate text-text-primary">
              {session.repo_id}
            </div>
            <div className="text-xs text-text-muted truncate mt-0.5">
              {getSessionExcerpt(session)}
            </div>
          </div>

          <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide ${badge.className}`}>
            {badge.label}
          </span>

          <div className="flex items-center gap-1.5 shrink-0">
            {/* Delete button — visible on hover, disabled while indexing */}
            <button
              onClick={handleDelete}
              type="button"
              title={isIndexing ? 'Cannot delete while indexing' : 'Delete session'}
              disabled={isIndexing}
              className={`opacity-0 group-hover:opacity-100 transition-all mt-0.5 ${
                isIndexing
                  ? 'text-text-muted cursor-not-allowed opacity-30'
                  : 'text-text-muted hover:text-offline'
              }`}
              aria-label="Delete session"
            >
              <TrashIcon />
            </button>
          </div>
        </div>

        <div className="text-2xs text-text-muted mt-1">
          {getRelativeTime(session.last_active || session.created_at)}
        </div>
      </div>

      {confirmOpen && (
        <ConfirmDialog
          message="Delete this repo session? This removes chat history, indexing metadata, and the associated vector index if safe. This cannot be undone."
          confirmLabel="Delete"
          onConfirm={() => {
            setConfirmOpen(false);
            onDelete(session.id);
          }}
          onCancel={() => setConfirmOpen(false)}
        />
      )}
    </>
  );
}


function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z" />
      <path
        fillRule="evenodd"
        d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"
      />
    </svg>
  );
}
