import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SourceCard from './SourceCard';
import { buildAnswerDiagnosticsRows } from './answerDiagnostics';
import { groupSources } from './sourceCards';

/**
 * Three bouncing dots for the loading state.
 */
function LoadingDots() {
  return (
    <div className="flex items-end gap-1 py-1 px-0.5 h-6">
      <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-dot-1" />
      <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-dot-2" />
      <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-dot-3" />
    </div>
  );
}

/**
 * Custom renderers for react-markdown — enforces our theme inside code blocks.
 */
const markdownComponents = {
  /* react-markdown v9: fenced code blocks are <pre><code>…</code></pre>.
     Inline backticks are just <code>…</code> inside <p> or <li>.
     We style `code` as inline by default, and the `pre` component wraps
     fenced blocks.  CSS `pre code` overrides reset the inline styles. */
  pre({ children, ...props }) {
    return (
      <pre className="bg-surface-3 border border-border rounded-lg p-2.5 my-2 overflow-x-auto" {...props}>
        {children}
      </pre>
    );
  },
  code({ className, children, ...props }) {
    return (
      <code
        className={`inline-code font-mono text-text-primary bg-surface-3 px-1.5 py-0.5 rounded-md text-[0.82em] border border-border ${className || ''}`}
        {...props}
      >
        {children}
      </code>
    );
  },
  a({ href, children }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="text-text-primary underline decoration-text-muted underline-offset-4 hover:decoration-text-secondary"
      >
        {children}
      </a>
    );
  },
  p({ children }) {
    return <p className="mb-1.5 last:mb-0 leading-[1.55] text-[0.88rem] text-text-primary/95">{children}</p>;
  },
  h1({ children }) {
    return <h1 className="text-[0.95rem] font-semibold text-text-primary mb-1.5">{children}</h1>;
  },
  h2({ children }) {
    return <h2 className="text-[0.88rem] font-semibold text-text-primary mt-2.5 mb-1">{children}</h2>;
  },
  h3({ children }) {
    return <h3 className="text-[0.84rem] font-medium text-text-primary mt-2 mb-0.5">{children}</h3>;
  },
  ul({ children }) {
    return <ul className="mb-1.5 space-y-0.5 pl-0">{children}</ul>;
  },
  ol({ children }) {
    return <ol className="mb-1.5 space-y-0.5 pl-0">{children}</ol>;
  },
  li({ children, ordered }) {
    return (
      <li className="flex items-start gap-1.5 text-[0.87rem] leading-[1.5] text-text-primary/92">
        <span className="mt-[0.62rem] h-1.5 w-1.5 shrink-0 rounded-full bg-text-muted" />
        <span className="min-w-0">{children}</span>
      </li>
    );
  },
  strong({ children }) {
    return <strong className="font-semibold text-text-primary">{children}</strong>;
  },
  blockquote({ children }) {
    return (
      <blockquote className="my-3 rounded-r-xl border-l-2 border-text-muted bg-surface-3/60 px-3 py-2 text-text-secondary">
        {children}
      </blockquote>
    );
  },
  hr() {
    return <hr className="my-4 border-0 border-t border-border" />;
  },
  table({ children }) {
    return (
      <div className="my-3 overflow-x-auto rounded-xl border border-border">
        <table className="w-full border-collapse text-left text-sm">{children}</table>
      </div>
    );
  },
  thead({ children }) {
    return <thead className="bg-surface-3 text-text-primary">{children}</thead>;
  },
  th({ children }) {
    return <th className="px-3 py-2 font-medium border-b border-border">{children}</th>;
  },
  td({ children }) {
    return <td className="px-3 py-2 align-top border-b border-border last:border-b-0">{children}</td>;
  },
};

export default function MessageBubble({ message, sessionId = '', userQuery = '' }) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [copiedUser, setCopiedUser] = useState(false);
  const [copiedDiagnostics, setCopiedDiagnostics] = useState(false);
  const diagnosticsRows = buildAnswerDiagnosticsRows(message.diagnostics);
  const sourcesRef = useRef(null);
  const diagnosticsRef = useRef(null);

  // Auto-close Sources/Diagnostics when scrolled out of view
  useEffect(() => {
    const refs = [sourcesRef, diagnosticsRef];
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting && entry.target.open) {
            entry.target.open = false;
          }
        });
      },
      { threshold: 0 }
    );
    refs.forEach((ref) => {
      if (ref.current) observer.observe(ref.current);
    });
    return () => observer.disconnect();
  }, []);

  const handleCopyDiagnostics = (e) => {
    e.stopPropagation();
    e.preventDefault();
    const lines = ['### Diagnostics'];
    const sections = ['Intent', 'Model', 'Sources', 'Validation', 'Freshness'];
    sections.forEach((sec) => {
      const secRows = diagnosticsRows.filter((r) => r.section === sec);
      if (secRows.length > 0) {
        lines.push(`\n#### ${sec}`);
        secRows.forEach((row) => {
          if (row.kind === 'list') {
            lines.push(`- ${row.label}:`);
            row.value.forEach((item) => {
              lines.push(`  - ${item}`);
            });
          } else {
            lines.push(`- ${row.label}: ${row.value}`);
          }
        });
      }
    });

    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      setCopiedDiagnostics(true);
      setTimeout(() => setCopiedDiagnostics(false), 1500);
    });
  };

  const handleCopyUser = () => {
    const text = typeof message.content === 'string' ? message.content.trim() : '';
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopiedUser(true);
      setTimeout(() => setCopiedUser(false), 1500);
    });
  };

  const handleCopyResponse = () => {
    const text = typeof message.content === 'string' ? message.content.trim() : '';
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  if (isUser) {
    return (
      <div className="flex justify-end animate-fadeIn group">
        <div className="max-w-[75%]">
          <div className="bg-surface-3 border border-border rounded-2xl px-4 py-3 text-text-primary text-sm whitespace-pre-wrap break-words">
            {message.content}
          </div>
          <div className="flex items-center justify-end gap-2 mt-1 pr-0.5">
            <button
              onClick={handleCopyUser}
              className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-3 px-2 py-0.5 font-mono text-2xs text-text-secondary opacity-0 group-hover:opacity-100 transition-all duration-150 hover:border-text-muted hover:text-text-primary"
              title="Copy prompt"
              aria-label="Copy prompt"
            >
              <CopyIcon />
              {copiedUser ? 'Copied' : 'Copy'}
            </button>
            <span className="text-2xs text-text-muted">
              {formatTimestamp(message.timestamp)}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Assistant — loading state (only if no content yet)
  if (message.loading && !message.content) {
    return (
      <div className="flex justify-start animate-fadeIn">
        <div className="max-w-[75%] bg-surface-2 border border-border rounded-2xl px-4 py-3">
          <LoadingDots />
        </div>
      </div>
    );
  }

  // Assistant — error state
  if (message.error) {
    return (
      <div className="flex justify-start animate-fadeIn">
        <div className="max-w-[75%]">
          <div className="bg-surface-2 border border-offline/30 rounded-2xl px-4 py-3 text-offline/80 text-sm">
            ⚠ {message.content}
          </div>
          <div className="text-2xs text-text-muted mt-1 pl-0.5">
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }

  // Assistant — normal answer
  return (
    <div className="flex justify-start animate-fadeIn group">
      <div className="max-w-[90%] min-w-0">
          <div className="px-1 py-1 text-text-primary">
            <div className="assistant-response max-w-none text-text-primary">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
            </div>
          </div>

          <div className="flex items-center gap-2 text-2xs text-text-muted mt-1 pl-0.5 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
              <button
                onClick={handleCopyResponse}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-3 px-2 py-0.5 font-mono text-text-secondary transition-colors hover:border-text-muted hover:text-text-primary"
                title="Copy response"
                aria-label="Copy response"
              >
                <CopyIcon />
                {copied ? 'Copied' : 'Copy'}
              </button>
              {message.context_tokens != null && (
                <span
                  className="rounded-full border border-border bg-surface-3 px-2 py-0.5 font-mono"
                  title="Context tokens used"
                >
                  {message.context_tokens} tok
                </span>
              )}
              <span>{formatTimestamp(message.timestamp)}</span>
          </div>

          {(diagnosticsRows.length > 0 || (message.sources && message.sources.length > 0)) && (
            <div className="mt-1 border-t border-border/40 pt-2 pl-1 space-y-1.5">
              {message.sources && message.sources.length > 0 && (
                <details className="group" ref={sourcesRef}>
                <summary className="flex cursor-pointer items-center justify-between gap-3 list-none outline-none select-none">
                  <div className="flex items-center gap-2 text-2xs text-text-muted uppercase tracking-[0.22em] font-medium transition-colors hover:text-text-secondary">
                    <svg
                      className="h-3 w-3 transform text-text-muted transition-transform duration-200 group-open:rotate-90"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    Sources
                  </div>
                  <div className="rounded-full border border-border bg-surface-3 px-2 py-0.5 text-2xs font-mono text-text-muted">
                    {(message.sources || []).length}
                  </div>
                </summary>
                <div className="mt-2 space-y-2.5 animate-fadeIn">
                  {(() => {
                    const groupedSources = groupSources(message.sources || []);
                    return Object.entries(groupedSources).map(([role, items]) => {
                      if (items.length === 0) return null;
                      return (
                        <div key={role} className="space-y-1.5">
                          <div className="text-[10px] font-mono uppercase tracking-wider text-text-muted select-none">
                            {role} ({items.length})
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {items.map((item, idx) => (
                              <SourceCard key={idx} source={item.original} />
                            ))}
                          </div>
                        </div>
                      );
                    });
                  })()}
                </div>
              </details>
              )}

              {diagnosticsRows.length > 0 && (
                <details className="group" ref={diagnosticsRef}>
                  <summary className="flex cursor-pointer items-center justify-between gap-3 list-none outline-none select-none">
                    <div className="flex items-center gap-2 text-2xs text-text-muted uppercase tracking-[0.22em] font-medium transition-colors hover:text-text-secondary">
                      <svg
                        className="h-3 w-3 transform text-text-muted transition-transform duration-200 group-open:rotate-90"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={3}
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                      Diagnostics
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleCopyDiagnostics}
                        className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-3 px-2 py-0.5 font-mono text-3xs text-text-secondary transition-colors hover:border-text-muted hover:text-text-primary"
                        title="Copy diagnostics info"
                        aria-label="Copy diagnostics info"
                      >
                        <CopyIcon />
                        {copiedDiagnostics ? 'Copied' : 'Copy'}
                      </button>
                      <div className="rounded-full border border-border bg-surface-3 px-2 py-0.5 text-2xs font-mono text-text-muted">
                        {diagnosticsRows.length}
                      </div>
                    </div>
                  </summary>
                  <div className="mt-3 space-y-4">
                    {['Intent', 'Model', 'Sources', 'Validation', 'Freshness'].map((sectionName) => {
                      const sectionRows = diagnosticsRows.filter((row) => row.section === sectionName);
                      if (sectionRows.length === 0) return null;

                      const basicRows = sectionRows.filter((row) => !row.isAdvanced);
                      const advancedRows = sectionRows.filter((row) => row.isAdvanced);

                      return (
                        <div key={sectionName} className="w-full rounded-xl border border-border bg-surface-2/40 p-3 space-y-2.5">
                          <div className="text-[10px] uppercase tracking-[0.22em] text-text-muted font-bold border-b border-border/40 pb-1">
                            {sectionName}
                          </div>
                          
                          {/* Basic fields */}
                          {basicRows.length > 0 && (
                            <div className="space-y-3">
                              {basicRows.map((row) => (
                                <div key={row.label} className="space-y-1">
                                  <div className="text-[10px] text-text-muted font-semibold">
                                    {row.label}
                                  </div>
                                  {row.kind === 'list' ? (
                                    <div className="flex flex-wrap gap-1.5">
                                      {row.value.map((item, index) => (
                                        <span
                                          key={`${row.label}-${index}`}
                                          className="rounded-full border border-border bg-surface-3 px-2 py-0.5 font-mono text-[10px] text-text-primary animate-fadeIn"
                                        >
                                          {item}
                                        </span>
                                      ))}
                                    </div>
                                  ) : (
                                    <div className="text-xs text-text-primary break-words font-mono">
                                      {row.value}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Advanced fields behind expandable sub-section */}
                          {advancedRows.length > 0 && (
                            <details className="group mt-2 pt-2 border-t border-border/30">
                              <summary className="flex cursor-pointer items-center gap-1.5 list-none outline-none select-none text-[10px] text-text-muted hover:text-text-secondary font-mono">
                                <svg
                                  className="h-2.5 w-2.5 transform text-text-muted transition-transform duration-200 group-open:rotate-90"
                                  fill="none"
                                  viewBox="0 0 24 24"
                                  stroke="currentColor"
                                  strokeWidth={3}
                                >
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                                </svg>
                                Advanced details
                              </summary>
                              <div className="mt-2.5 space-y-3 pl-2.5 border-l border-border/40">
                                {advancedRows.map((row) => (
                                  <div key={row.label} className="space-y-1">
                                    <div className="text-[10px] text-text-muted font-semibold">
                                      {row.label}
                                    </div>
                                    {row.kind === 'list' ? (
                                      <div className="flex flex-wrap gap-1.5">
                                        {row.value.map((item, index) => (
                                          <span
                                            key={`${row.label}-${index}`}
                                            className="rounded-full border border-border bg-surface-3 px-2 py-0.5 font-mono text-[10px] text-text-primary animate-fadeIn"
                                          >
                                            {item}
                                          </span>
                                        ))}
                                      </div>
                                    ) : (
                                      <div className="text-xs text-text-primary break-words font-mono">
                                        {row.value}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </details>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </details>
              )}
            </div>
          )}
      </div>
    </div>
  );
}

function formatTimestamp(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M4 2.5A1.5 1.5 0 0 1 5.5 1h6A1.5 1.5 0 0 1 13 2.5v8A1.5 1.5 0 0 1 11.5 12h-6A1.5 1.5 0 0 1 4 10.5v-8zm1.5-.5a.5.5 0 0 0-.5.5v8a.5.5 0 0 0 .5.5h6a.5.5 0 0 0 .5-.5v-8a.5.5 0 0 0-.5-.5h-6z" />
      <path d="M2.5 4A1.5 1.5 0 0 0 1 5.5v8A1.5 1.5 0 0 0 2.5 15h6a1.5 1.5 0 0 0 1.415-1H8.5a2.5 2.5 0 0 1-2.5-2.5V4H2.5z" />
    </svg>
  );
}
