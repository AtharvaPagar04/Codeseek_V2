import React from 'react';
import { fetchGraphCodeBlock } from '../../utils/api';
import { languageLabel, lineCount, normalizeLanguage, tokenClass, tokenizeCodeLine } from './codeHighlight';

const codeBlockCache = new Map();

function lineLabel(block) {
  const start = block?.start_line;
  const end = block?.end_line;
  if (start && end && start !== end) return `Lines ${start}-${end}`;
  if (start) return `Line ${start}`;
  return '';
}

export default function GraphCodePreviewModal({ sessionId, codeBlock, onClose }) {
  const chunkId = codeBlock?.chunk_id || '';
  const cacheKey = `${sessionId}:${chunkId}`;
  const [state, setState] = React.useState({
    status: chunkId ? 'loading' : 'unavailable',
    data: null,
    error: '',
  });
  const [copied, setCopied] = React.useState(false);

  React.useEffect(() => {
    if (!chunkId || !sessionId) {
      setState({ status: 'unavailable', data: null, error: 'Code preview unavailable for this block.' });
      return undefined;
    }
    const cached = codeBlockCache.get(cacheKey);
    if (cached) {
      setState({ status: 'ready', data: cached, error: '' });
      return undefined;
    }

    let cancelled = false;
    setState({ status: 'loading', data: null, error: '' });
    fetchGraphCodeBlock(sessionId, chunkId)
      .then((data) => {
        if (cancelled) return;
        if (!data || data.status === 'not_found') {
          setState({
            status: 'unavailable',
            data,
            error: data?.message || 'Code block source is not available for this chunk.',
          });
          return;
        }
        codeBlockCache.set(cacheKey, data);
        setState({ status: 'ready', data, error: '' });
      })
      .catch((error) => {
        if (cancelled) return;
        setState({ status: 'error', data: null, error: error?.message || 'Unable to load code preview.' });
      });
    return () => {
      cancelled = true;
    };
  }, [cacheKey, chunkId, sessionId]);

  React.useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose?.();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!codeBlock) return null;

  const data = state.data || {};
  const title = data.name || codeBlock.displayName || codeBlock.name || 'Code block';
  const path = data.path || codeBlock.path || '';
  const lines = lineLabel(data) || codeBlock.lineLabel || '';
  const description = data.description || codeBlock.displayDescription || codeBlock.description || '';
  const language = data.language || codeBlock.language || '';
  const code = data.code || '';
  const normalizedLanguage = normalizeLanguage(language, path);
  const displayLanguage = languageLabel(language, path);
  const codeLines = code ? code.split('\n') : [];
  const firstLineNumber = Number(data.start_line || codeBlock.start_line || 1) || 1;

  const handleCopy = async () => {
    if (!code || typeof navigator === 'undefined' || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-3 backdrop-blur-sm sm:p-6"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Code preview"
    >
      <div
        className="flex h-[88vh] w-[95vw] max-w-[1180px] flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0b0b0d] shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-white/10 px-5 py-3">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-wider text-text-muted font-bold">Code Preview</div>
            <h3 className="mt-1 break-words font-mono text-sm font-semibold text-text-primary">{title}</h3>
            {path && <div className="mt-1 break-all font-mono text-[10px] text-text-muted">{path}</div>}
            <div className="mt-2 flex flex-wrap gap-1.5">
              <span className="inline-flex items-center gap-1 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[9px] uppercase tracking-wide text-cyan-200">
                <CodeIcon className="h-3 w-3" />
                {displayLanguage}
              </span>
              {lines && (
                <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[9px] text-text-muted">
                  {lines}
                </span>
              )}
              {data.truncated && (
                <span className="rounded-full border border-warning/40 bg-warning/10 px-2 py-0.5 text-[9px] text-warning">
                  Truncated to {data.max_lines} lines
                </span>
              )}
            </div>
            {description && (
              <p className="mt-2 line-clamp-2 max-w-4xl text-[11px] leading-relaxed text-text-muted">{description}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close code preview"
            className="shrink-0 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 font-mono text-[10px] text-text-muted transition-colors hover:border-white/30 hover:text-text-primary"
          >
            Close
          </button>
        </div>

        <div className="flex items-center justify-between border-b border-white/10 bg-[#050505] px-5 py-2">
          <div className="flex items-center gap-2 font-mono text-[10px] text-text-muted">
            <CodeIcon className="h-3.5 w-3.5 text-cyan-300" />
            <span className="text-cyan-200">{displayLanguage}</span>
            <span className="text-slate-600">/</span>
            <span>{state.status === 'ready' ? `${lineCount(code)} lines` : state.status}</span>
            {normalizedLanguage !== 'text' && <span className="text-slate-600">.{normalizedLanguage}</span>}
          </div>
          <button
            type="button"
            onClick={handleCopy}
            disabled={!code}
            aria-label={copied ? 'Code copied' : 'Copy code'}
            title={copied ? 'Copied' : 'Copy code'}
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 font-mono text-[10px] text-text-muted transition-colors hover:border-cyan-500/40 hover:text-cyan-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <CopyIcon className="h-3.5 w-3.5" />
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto bg-[#050505]">
          {state.status === 'loading' && (
            <div className="p-5 font-mono text-[11px] text-text-muted">Loading code preview...</div>
          )}
          {(state.status === 'error' || state.status === 'unavailable') && (
            <div className="p-5 text-[12px] text-text-muted">{state.error || 'Code preview unavailable.'}</div>
          )}
          {state.status === 'ready' && (
            <pre className="m-0 min-w-max p-5 font-mono text-[13px] leading-[1.7]">
              <code>
                {codeLines.map((line, lineIndex) => (
                  <span key={lineIndex} className="flex min-h-[1.7em]">
                    <span className="mr-5 w-10 shrink-0 select-none text-right text-slate-700">
                      {firstLineNumber + lineIndex}
                    </span>
                    <span className="whitespace-pre">
                      {tokenizeCodeLine(line).map((token, tokenIndex) => (
                        <span key={`${lineIndex}-${tokenIndex}`} className={tokenClass(token.type)}>
                          {token.text}
                        </span>
                      ))}
                    </span>
                  </span>
                ))}
              </code>
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function CodeIcon({ className = '' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M8 8l-4 4 4 4M16 8l4 4-4 4M14 5l-4 14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CopyIcon({ className = '' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="9" y="9" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <path d="M5 15V7a2 2 0 0 1 2-2h8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
