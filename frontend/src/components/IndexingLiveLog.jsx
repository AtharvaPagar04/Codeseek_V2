import React from 'react';
import { useState, useEffect, useRef } from 'react';

const API_BASE = import.meta.env?.VITE_API_BASE_URL?.replace(/\/$/, "") || 'http://127.0.0.1:8000';

/**
 * IndexingLiveLog — shows a real-time activity log during session indexing.
 *
 * Props:
 *   session         — the session being indexed
 *   onRetryIndexing — function callback to trigger indexing retry
 *   onCancelIndexing — function callback to trigger indexing cancellation
 */
export default function IndexingLiveLog({ session, onRetryIndexing, onCancelIndexing }) {
  const sessionId = session?.id;
  const isIndexing = session?.status === 'indexing';
  const [events, setEvents] = useState([]);
  const [sseStatus, setSseStatus] = useState('idle'); // idle | connected | disconnected
  const bottomRef = useRef(null);
  const sseRef = useRef(null);
  const retryTimer = useRef(null);

  const [visible, setVisible] = useState(isIndexing || session?.status === 'failed');
  const wasIndexingRef = useRef(isIndexing);

  useEffect(() => {
    if (isIndexing) {
      setVisible(true);
      wasIndexingRef.current = true;
    } else if (session?.status === 'failed') {
      setVisible(true);
      wasIndexingRef.current = false;
    } else if (session?.status === 'ready') {
      if (wasIndexingRef.current) {
        const timer = setTimeout(() => {
          setVisible(false);
          wasIndexingRef.current = false;
        }, 5000);
        return () => clearTimeout(timer);
      } else {
        setVisible(false);
      }
    }
  }, [isIndexing, session?.status]);

  useEffect(() => {
    setVisible(isIndexing || session?.status === 'failed');
    wasIndexingRef.current = isIndexing;
  }, [sessionId]);

  // Fetch existing events on mount / when session changes.
  useEffect(() => {
    if (!sessionId) return;
    fetch(`${API_BASE}/api/v1/sessions/${sessionId}/indexing-events`, {
      credentials: 'include',
    })
      .then((res) => (res.ok ? res.json() : { events: [] }))
      .then((data) => {
        if (data.events?.length) {
          setEvents((prev) => dedup([...prev, ...data.events]));
        }
      })
      .catch(() => {});
  }, [sessionId]);

  // SSE subscription while indexing.
  useEffect(() => {
    if (!sessionId || !isIndexing) {
      closeSse();
      return;
    }
    openSse();
    return () => closeSse();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, isIndexing]);

  function openSse() {
    closeSse();
    const ctrl = new AbortController();
    sseRef.current = ctrl;
    setSseStatus('connected');

    (async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/sessions/${sessionId}/indexing-events/stream`,
          { credentials: 'include', signal: ctrl.signal },
        );
        if (!res.ok || !res.body) {
          setSseStatus('disconnected');
          scheduleRetry();
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const evt = JSON.parse(line.slice(6));
                setEvents((prev) => dedup([...prev, evt]));
              } catch {
                // ignore
              }
            }
          }
        }
        setSseStatus('idle');
      } catch (err) {
        if (err.name === 'AbortError') return;
        setSseStatus('disconnected');
        scheduleRetry();
      }
    })();
  }

  function closeSse() {
    sseRef.current?.abort();
    sseRef.current = null;
    clearTimeout(retryTimer.current);
  }

  function scheduleRetry() {
    clearTimeout(retryTimer.current);
    retryTimer.current = setTimeout(() => {
      if (isIndexing) openSse();
    }, 2000);
  }

  // Auto-scroll on new events.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  if (!visible) return null;
  if (events.length === 0 && !isIndexing && session?.status !== 'failed') return null;

  const latest = events[events.length - 1];
  const terminalStage = latest?.stage === 'complete' || latest?.stage === 'failed';

  return (
    <div
      className="w-full max-w-xl mb-4 rounded-xl border border-border bg-surface-2/60 overflow-hidden shadow-lg animate-fadeIn animate-duration-300"
      style={{ backdropFilter: 'blur(6px)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3.5 py-1.5 border-b border-border/50 bg-surface-2/40">
        <span className="text-[10px] font-mono font-medium text-text-secondary tracking-wide uppercase">
          {session?.status === 'failed' ? 'Indexing Log (Failed)' : isIndexing ? 'Indexing…' : 'Indexing Log'}
        </span>
        <StatusDot status={session?.status === 'failed' ? 'error' : latest?.level || 'info'} />
      </div>

      {/* Event log */}
      {events.length > 0 && (
        <div className="max-h-24 overflow-y-auto px-3.5 py-2 space-y-1 scrollbar-thin">
          {events.map((evt) => (
            <EventLine key={evt.id} event={evt} />
          ))}
          <div ref={bottomRef} />
        </div>
      )}

      {/* Progress bar for latest progress */}
      {latest?.progress != null && latest?.total > 0 && !terminalStage && isIndexing && (
        <div className="px-3.5 pb-2">
          <div className="w-full h-1 rounded-full bg-surface-3 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${Math.min(100, Math.round((latest.progress / latest.total) * 100))}%`,
                background: 'linear-gradient(90deg, #22c55e, #3b82f6)',
              }}
            />
          </div>
          <div className="text-right text-[9px] text-text-muted mt-0.5 font-mono">
            {latest.progress}/{latest.total}
          </div>
        </div>
      )}

      {/* SSE disconnected notice */}
      {sseStatus === 'disconnected' && isIndexing && (
        <div className="px-3.5 pb-2 text-[9px] text-warning font-mono">
          Live updates disconnected. Retrying…
        </div>
      )}

      {/* Inline status message notice */}
      {session?.status === 'indexing' && (
        <div className="border-t border-border/30 bg-warning/5 px-3.5 py-1.5 text-[10px] text-warning font-sans flex items-center justify-between gap-1.5">
          <div className="flex items-center gap-1.5">
            <svg className="w-3 h-3 shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>Indexing in progress. Chat will enable once complete.</span>
          </div>
          {onCancelIndexing && (
            <button
              onClick={() => onCancelIndexing(session.id)}
              className="shrink-0 px-2 py-0.5 rounded bg-warning/80 hover:bg-warning text-surface-1 font-semibold text-[9px] transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      )}

      {session?.status === 'failed' && (
        <div className="border-t border-border/30 bg-offline/5 px-3.5 py-2 text-[10px] text-offline font-sans flex items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 min-w-0">
            <svg className="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="min-w-0 truncate">
              {session.error ? `Failed: ${session.error}` : 'Indexing failed.'}
            </span>
          </div>
          {onRetryIndexing && (
            <button
              onClick={() => onRetryIndexing(session.id)}
              className="shrink-0 px-2 py-0.5 rounded bg-offline text-surface-1 font-semibold text-[9px] hover:opacity-90 transition-opacity"
            >
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function EventLine({ event }) {
  const icon = levelIcon(event.level);
  const color = levelColor(event.level);
  return (
    <div className="flex items-start gap-1.5 text-[10px] leading-relaxed font-mono">
      <span className={`shrink-0 mt-0.5 ${color}`}>{icon}</span>
      <span className="text-text-secondary">{event.message}</span>
    </div>
  );
}

function StatusDot({ status }) {
  const bg =
    status === 'success'
      ? 'bg-online'
      : status === 'error'
        ? 'bg-offline'
        : status === 'warning'
          ? 'bg-warning'
          : 'bg-text-muted';
  const pulse = status === 'info' ? 'animate-pulse' : '';
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${bg} ${pulse}`} />;
}

function levelIcon(level) {
  if (level === 'success') return '✓';
  if (level === 'error') return '✗';
  if (level === 'warning') return '⚠';
  return '•';
}

function levelColor(level) {
  if (level === 'success') return 'text-online';
  if (level === 'error') return 'text-offline';
  if (level === 'warning') return 'text-warning';
  return 'text-text-muted';
}

function dedup(events) {
  const seen = new Set();
  const out = [];
  for (const e of events) {
    if (seen.has(e.id)) continue;
    seen.add(e.id);
    out.push(e);
  }
  return out.sort((a, b) => a.id - b.id);
}
