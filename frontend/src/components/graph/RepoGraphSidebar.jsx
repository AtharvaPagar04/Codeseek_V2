import React from 'react';
import { NODE_COLORS } from './graphTheme';

const MODE_LABELS = {
  imports: 'Imports',
  structure: 'Structure',
  retrieval_trace: 'Retrieval Trace',
};

const NODE_TYPE_LABELS = {
  folder: 'Folders',
  file: 'Files',
  symbol: 'Symbols',
  external: 'External',
};

const TRACE_NODE_TYPE_LABELS = {
  query: 'Query',
  chunk: 'Chunks',
  answer: 'Answer',
};

const EDGE_TYPE_LABELS = {
  contains: 'Contains',
  imports: 'Imports',
  defines: 'Defines',
  retrieval: 'Retrieval',
};

const TRACE_EDGE_TYPE_LABELS = {
  retrieved: 'Retrieved',
  graph_added: 'Graph-added',
  reranked: 'Reranked in',
  context_selected: 'Context',
  final_context: 'Final context',
  cited: 'Cited',
  dropped: 'Dropped',
};

export default function RepoGraphSidebar({
  mode,
  modes,
  onModeChange,
  traceSourceMode = 'latest',
  selectedTraceMessageId = null,
  traceMessages = [],
  traceMessagesStatus = 'idle',
  traceMessagesError = null,
  onTraceSourceChange,
  searchQuery,
  onSearchChange,
  nodeTypeFilter,
  onNodeTypeFilterChange,
  edgeTypeFilter,
  onEdgeTypeFilterChange,
  summary,
  onResetView,
  onFitView,
}) {
  const isTraceMode = mode === 'retrieval_trace';
  const nodeTypeLabels = isTraceMode ? TRACE_NODE_TYPE_LABELS : NODE_TYPE_LABELS;
  const edgeTypeLabels = isTraceMode ? TRACE_EDGE_TYPE_LABELS : EDGE_TYPE_LABELS;

  const toggleNodeType = (type) => {
    if (nodeTypeFilter.includes(type)) {
      onNodeTypeFilterChange(nodeTypeFilter.filter((t) => t !== type));
    } else {
      onNodeTypeFilterChange([...nodeTypeFilter, type]);
    }
  };

  const toggleEdgeType = (type) => {
    if (edgeTypeFilter.includes(type)) {
      onEdgeTypeFilterChange(edgeTypeFilter.filter((t) => t !== type));
    } else {
      onEdgeTypeFilterChange([...edgeTypeFilter, type]);
    }
  };

  const setTraceFilter = (types) => {
    onEdgeTypeFilterChange(types);
  };

  const traceStageCount = (type) => {
    if (!isTraceMode) return summary.edge_types?.[type] || 0;
    if (type === 'retrieved') return summary.retrieved_count || 0;
    if (type === 'graph_added') return summary.graph_added_count || 0;
    if (type === 'reranked') return summary.reranked_count || 0;
    if (type === 'context_selected') return summary.context_selected_count || 0;
    if (type === 'final_context') return summary.final_source_count ?? summary.final_count ?? 0;
    if (type === 'cited') return summary.cited_count || 0;
    if (type === 'dropped') return summary.dropped_count || 0;
    return summary.edge_types?.[type] || 0;
  };

  const selectedTraceItem = traceMessages.find(
    (item) => item.assistant_message_id === selectedTraceMessageId
  );

  return (
    <div className="flex flex-col gap-4 w-[220px] shrink-0 overflow-y-auto p-4 border-r border-border bg-surface-2/40 backdrop-blur-md">
      {/* Header */}
      <div>
        <h2 className="text-sm font-semibold text-text-primary font-mono tracking-wide">Repo Graph</h2>
        <p className="text-[10px] text-text-muted mt-0.5 leading-tight">
          {isTraceMode
            ? 'Explain how the latest answer was assembled.'
            : 'Visualize files, symbols, imports, and retrieval paths.'}
        </p>
      </div>

      {/* Mode selector */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">Mode</div>
        <div className="flex flex-col gap-1">
          {modes.map((m) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              className={`text-left px-2.5 py-1.5 rounded-lg text-[10px] font-mono transition-colors ${
                mode === m
                  ? 'bg-surface-3 border border-text-muted text-text-primary'
                  : 'text-text-muted hover:text-text-primary hover:bg-surface-3/50'
              }`}
            >
              {MODE_LABELS[m] || m}
            </button>
          ))}
        </div>
      </div>

      {/* Search */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">Search</div>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search files, symbols, paths…"
          className="w-full px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-primary placeholder-text-muted focus:outline-none focus:border-text-muted transition-colors"
        />
      </div>

      {isTraceMode && (
        <div className="rounded-xl border border-border bg-surface-3/45 p-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">Viewing trace for</div>
              <div className="mt-0.5 text-[10px] font-mono text-text-primary">
                {traceSourceMode === 'message' && selectedTraceMessageId ? 'Selected answer' : 'Latest answer'}
              </div>
            </div>
            <button
              type="button"
              onClick={() => onTraceSourceChange?.({ sourceMode: 'latest', assistantMessageId: null })}
              className={`rounded-full border px-2 py-0.5 text-[9px] font-mono transition-colors ${
                traceSourceMode === 'latest'
                  ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-100'
                  : 'border-border bg-surface-2 text-text-muted hover:text-text-primary'
              }`}
            >
              Latest
            </button>
          </div>

          {selectedTraceItem && (
            <p className="mt-2 line-clamp-2 text-[9px] leading-relaxed text-text-secondary">
              {selectedTraceItem.answer_preview || selectedTraceItem.assistant_message_id}
            </p>
          )}

          <div className="mt-3 space-y-1.5">
            <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">Recent answers</div>
            {traceMessagesStatus === 'loading' && (
              <div className="space-y-1">
                {[0, 1, 2].map((idx) => (
                  <div key={idx} className="h-12 animate-pulse rounded-lg border border-border bg-surface-2/70" />
                ))}
              </div>
            )}
            {traceMessagesStatus === 'error' && (
              <p className="rounded-lg border border-offline/20 bg-offline/10 px-2 py-1.5 text-[9px] text-offline/80">
                {traceMessagesError || 'Trace messages unavailable.'}
              </p>
            )}
            {traceMessagesStatus !== 'loading' && traceMessages.length === 0 && (
              <p className="rounded-lg border border-border bg-surface-2/70 px-2 py-1.5 text-[9px] leading-relaxed text-text-muted">
                No retrieval traces yet. Ask a question in chat first.
              </p>
            )}
            {traceMessages.slice(0, 8).map((item) => {
              const selected = traceSourceMode === 'message' && item.assistant_message_id === selectedTraceMessageId;
              const counts = item.summary || {};
              return (
                <button
                  key={item.assistant_message_id}
                  type="button"
                  onClick={() => onTraceSourceChange?.({ sourceMode: 'message', assistantMessageId: item.assistant_message_id })}
                  className={`w-full rounded-lg border px-2 py-2 text-left transition-colors ${
                    selected
                      ? 'border-cyan-500/35 bg-cyan-500/10'
                      : 'border-border bg-surface-2/70 hover:border-text-muted'
                  } ${item.trace_available ? '' : 'opacity-75'}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[9px] font-mono text-text-secondary">
                      {formatTraceTimestamp(item.created_at)}
                    </span>
                    <span
                      className={`rounded-full border px-1.5 py-0.5 text-[8px] font-mono ${
                        item.trace_version === 'v2'
                          ? 'border-emerald-500/25 bg-emerald-500/10 text-emerald-100'
                          : item.trace_available
                            ? 'border-amber-500/25 bg-amber-500/10 text-amber-100'
                            : 'border-border bg-surface-3 text-text-muted'
                      }`}
                    >
                      {item.trace_available ? (item.trace_version === 'v2' ? 'V2' : 'Fallback') : 'No trace'}
                    </span>
                  </div>
                  <div className="mt-1 line-clamp-2 text-[9px] leading-relaxed text-text-primary">
                    {item.answer_preview || item.assistant_message_id}
                  </div>
                  {item.trace_available && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      <TraceMiniBadge label="G" value={counts.graph_added_count || 0} />
                      <TraceMiniBadge label="F" value={counts.final_count || counts.final_source_count || 0} />
                      <TraceMiniBadge label="C" value={counts.cited_count || 0} />
                      {item.partial && <TraceMiniBadge label="Partial" />}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {isTraceMode && (
        <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/10 px-3 py-2">
          <div className="text-[9px] uppercase tracking-wider text-cyan-200 font-bold">Trace status</div>
          <div className="mt-1 text-[10px] text-text-primary font-mono capitalize">{summary.status || 'empty'}</div>
          <div className="mt-1 flex flex-wrap gap-1">
            <span className="rounded-full border border-cyan-500/25 bg-cyan-500/10 px-2 py-0.5 text-[8px] text-cyan-100 font-mono">
              {summary.trace_version || 'trace'}
            </span>
            {summary.partial && (
              <span className="rounded-full border border-amber-500/25 bg-amber-500/10 px-2 py-0.5 text-[8px] text-amber-100 font-mono">
                Partial trace
              </span>
            )}
          </div>
          {summary.message && (
            <p className="mt-1 text-[9px] leading-relaxed text-text-muted">{summary.message}</p>
          )}
          {summary.partial_reason && (
            <p className="mt-1 text-[9px] leading-relaxed text-amber-100/80">{summary.partial_reason}</p>
          )}
        </div>
      )}

      {isTraceMode && (
        <div>
          <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">Pipeline</div>
          <div className="grid grid-cols-2 gap-1.5">
            {[
              ['retrieved', 'Retrieved'],
              ['graph_added', 'Graph'],
              ['reranked', 'Reranked'],
              ['context_selected', 'Context'],
              ['final_context', 'Final'],
              ['cited', 'Cited'],
              ['dropped', 'Dropped'],
            ].map(([type, label]) => {
              const active = edgeTypeFilter.length === 0 || edgeTypeFilter.includes(type);
              return (
                <button
                  key={type}
                  type="button"
                  onClick={() => setTraceFilter([type])}
                  className={`rounded-lg border px-2 py-1.5 text-left transition-colors ${
                    active
                      ? 'border-cyan-500/30 bg-surface-3 text-text-primary'
                      : 'border-border bg-surface-3/50 text-text-muted'
                  }`}
                >
                  <div className="text-[8px] uppercase tracking-wide text-text-muted">{label}</div>
                  <div className="text-sm font-semibold font-mono">{traceStageCount(type)}</div>
                </button>
              );
            })}
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-1">
            <button
              type="button"
              onClick={() => setTraceFilter(['context_selected', 'final_context', 'cited'])}
              className="rounded-lg border border-border bg-surface-3/70 px-2 py-1 text-[9px] text-text-muted hover:text-text-primary"
            >
              Winners only
            </button>
            <button
              type="button"
              onClick={() => setTraceFilter(['dropped'])}
              className="rounded-lg border border-border bg-surface-3/70 px-2 py-1 text-[9px] text-text-muted hover:text-text-primary"
            >
              Dropped
            </button>
            <button
              type="button"
              onClick={() => setTraceFilter(['graph_added'])}
              className="rounded-lg border border-border bg-surface-3/70 px-2 py-1 text-[9px] text-text-muted hover:text-text-primary"
            >
              Graph only
            </button>
            <button
              type="button"
              onClick={() => setTraceFilter(['cited'])}
              className="rounded-lg border border-border bg-surface-3/70 px-2 py-1 text-[9px] text-text-muted hover:text-text-primary"
            >
              Cited only
            </button>
          </div>
        </div>
      )}

      {/* Node type filters */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">Node Types</div>
        <div className="flex flex-col gap-1">
          {Object.entries(nodeTypeLabels).map(([type, label]) => {
            const count = summary.node_types?.[type] || 0;
            const active = nodeTypeFilter.length === 0 || nodeTypeFilter.includes(type);
            return (
              <button
                key={type}
                onClick={() => toggleNodeType(type)}
                className={`flex items-center gap-2 px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                  active ? 'text-text-primary' : 'text-text-muted opacity-50'
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: NODE_COLORS[type] || NODE_COLORS.unknown }}
                />
                <span className="flex-1 text-left">{label}</span>
                <span className="text-text-muted text-[9px]">{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Edge type filters */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">
          {isTraceMode ? 'Stage Filters' : 'Edge Types'}
        </div>
        <div className="flex flex-col gap-1">
          {Object.entries(edgeTypeLabels).map(([type, label]) => {
            const count = isTraceMode ? traceStageCount(type) : (summary.edge_types?.[type] || 0);
            const active = edgeTypeFilter.length === 0 || edgeTypeFilter.includes(type);
            return (
              <button
                key={type}
                onClick={() => toggleEdgeType(type)}
                className={`flex items-center gap-2 px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                  active ? 'text-text-primary' : 'text-text-muted opacity-50'
                }`}
              >
                <span className="w-3 h-[2px] shrink-0 rounded" style={{ backgroundColor: active ? '#a78bfa' : '#555' }} />
                <span className="flex-1 text-left">{label}</span>
                <span className="text-text-muted text-[9px]">{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Stats */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">Stats</div>
        <div className="grid grid-cols-2 gap-1.5">
          {(isTraceMode
            ? [
                { label: 'Retrieved', value: summary.retrieved_count || 0 },
                { label: 'Graph', value: summary.graph_added_count || 0 },
                { label: 'Final', value: summary.final_count || 0 },
                { label: 'Cited', value: summary.cited_count || 0 },
              ]
            : [
                { label: 'Nodes', value: summary.node_count },
                { label: 'Edges', value: summary.edge_count },
                { label: 'Files', value: summary.node_types?.file || 0 },
                { label: 'Symbols', value: summary.node_types?.symbol || 0 },
              ]).map(({ label, value }) => (
            <div key={label} className="bg-surface-3 rounded-lg px-2 py-1.5 border border-border">
              <div className="text-[9px] text-text-muted">{label}</div>
              <div className="text-sm font-semibold text-text-primary font-mono">{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-1.5 mt-auto pt-3 border-t border-border">
        <button
          onClick={onFitView}
          className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors"
        >
          Fit to view
        </button>
        <button
          onClick={onResetView}
          className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors"
        >
          Reset filters
        </button>
      </div>

      {/* Legend */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold mb-1.5">Legend</div>
        <div className="flex flex-col gap-1">
          {isTraceMode ? (
            <>
              {[
                ['query', 'Query'],
                ['chunk', 'Evidence chunk'],
                ['answer', 'Answer'],
              ].map(([type, label]) => (
                <div key={type} className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: NODE_COLORS[type] || NODE_COLORS.unknown }} />
                  <span>{label}</span>
                </div>
              ))}
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono mt-1">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#4ade80', boxShadow: '0 0 4px #4ade80' }} />
                <span>Graph-added</span>
              </div>
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#2dd4bf', boxShadow: '0 0 4px #2dd4bf' }} />
                <span>Context</span>
              </div>
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#60a5fa', boxShadow: '0 0 4px #60a5fa' }} />
                <span>Final</span>
              </div>
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#f8fafc', boxShadow: '0 0 4px #f8fafc' }} />
                <span>Cited</span>
              </div>
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#ef4444', boxShadow: '0 0 4px #fb7185' }} />
                <span>Dropped</span>
              </div>
            </>
          ) : (
            <>
              {Object.entries(NODE_COLORS).filter(([k]) => !['unknown', 'query', 'chunk', 'answer'].includes(k)).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                  <span className="capitalize">{type}</span>
                </div>
              ))}
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono mt-1">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#22d3ee', boxShadow: '0 0 4px #22d3ee' }} />
                <span>Retrieved</span>
              </div>
              <div className="flex items-center gap-2 text-[9px] text-text-muted font-mono">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#4ade80', boxShadow: '0 0 4px #4ade80' }} />
                <span>Graph Assist</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function TraceMiniBadge({ label, value = null }) {
  return (
    <span className="rounded-full border border-border bg-surface-3 px-1.5 py-0.5 text-[8px] font-mono text-text-muted">
      {value === null ? label : `${label}:${value}`}
    </span>
  );
}

function formatTraceTimestamp(value) {
  if (!value) return 'answer';
  try {
    return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return 'answer';
  }
}
