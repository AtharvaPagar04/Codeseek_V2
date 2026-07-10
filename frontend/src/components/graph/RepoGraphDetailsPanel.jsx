import React from 'react';
import { fetchGraphNodeDetails } from '../../utils/api';
import { getNodeDegree, getNeighborIds } from './graphTransform';
import { NODE_COLORS } from './graphTheme';
import { buildNodeDetailsRenderModel, isFileNode as isGraphFileNode } from './nodeDetailsFormat';

const PANEL_CLASS = "relative z-30 flex w-[360px] min-w-[360px] max-w-[360px] shrink-0 flex-col gap-3 overflow-y-auto border-l border-border bg-surface-2 p-4 shadow-2xl";

export default function RepoGraphDetailsPanel({
  sessionId,
  selectedNode,
  nodes = [],
  links = [],
  mode,
  onFocusNeighbors,
  onResetFocus,
  onAskAbout,
  onCodeBlockSelect,
  onClose,
}) {
  const selectedNodeId = selectedNode?.id || '';
  const isFileNode = isGraphFileNode(selectedNode);

  const detailsCacheRef = React.useRef(new Map());
  const [detailsState, setDetailsState] = React.useState({
    key: '',
    status: 'idle',
    data: null,
    error: '',
  });

  React.useEffect(() => {
    detailsCacheRef.current.clear();
  }, [sessionId]);

  React.useEffect(() => {
    if (!sessionId || !selectedNodeId || !isFileNode) {
      setDetailsState({ key: '', status: 'idle', data: null, error: '' });
      return undefined;
    }

    const cacheKey = `${sessionId}:${selectedNodeId}`;
    const cached = detailsCacheRef.current.get(cacheKey);
    if (cached) {
      setDetailsState({ key: cacheKey, status: 'ready', data: cached, error: '' });
      return undefined;
    }

    let cancelled = false;
    setDetailsState({ key: cacheKey, status: 'loading', data: null, error: '' });
    fetchGraphNodeDetails(sessionId, selectedNodeId)
      .then((data) => {
        if (cancelled) return;
        if (!data) {
          setDetailsState({ key: cacheKey, status: 'empty', data: null, error: 'Details unavailable for this node.' });
          return;
        }
        detailsCacheRef.current.set(cacheKey, data);
        setDetailsState({ key: cacheKey, status: 'ready', data, error: '' });
      })
      .catch((error) => {
        if (cancelled) return;
        setDetailsState({
          key: cacheKey,
          status: 'error',
          data: null,
          error: error?.message || 'Unable to load graph node details.',
        });
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId, selectedNodeId, isFileNode]);

  const { inbound, outbound } = getNodeDegree(selectedNodeId, links);
  const neighbors = getNeighborIds(selectedNodeId, links);

  // Map nodes by id for quick lookup
  const nodesById = React.useMemo(() => {
    const map = new Map();
    if (nodes) {
      nodes.forEach((n) => map.set(n.id, n));
    }
    return map;
  }, [nodes]);

  const getLinkId = (nodeRef) => {
    if (typeof nodeRef === 'object' && nodeRef !== null) {
      return nodeRef.id;
    }
    return nodeRef;
  };

  // Group neighbors by connection type
  const categorizedNeighbors = React.useMemo(() => {
    const categories = {
      parents: [],      // inbound 'contains' or 'defines'
      children: [],     // outbound 'contains' or 'defines'
      imports: [],      // outbound 'imports'
      importedBy: [],   // inbound 'imports'
      others: [],       // everything else
    };

    if (!selectedNode || !links) return categories;

    const visited = new Set();

    links.forEach((link) => {
      const srcId = getLinkId(link.source);
      const tgtId = getLinkId(link.target);
      const type = link.type || '';

      if (srcId === selectedNode.id) {
        const neighbor = nodesById.get(tgtId);
        if (neighbor && !visited.has(tgtId)) {
          visited.add(tgtId);
          if (type === 'contains' || type === 'defines') {
            categories.children.push({ node: neighbor, type });
          } else if (type === 'imports') {
            categories.importedBy.push({ node: neighbor, type }); // Swapped because direction is reversed
          } else {
            categories.others.push({ node: neighbor, type, direction: 'outbound' });
          }
        }
      } else if (tgtId === selectedNode.id) {
        const neighbor = nodesById.get(srcId);
        if (neighbor && !visited.has(srcId)) {
          visited.add(srcId);
          if (type === 'contains' || type === 'defines') {
            categories.parents.push({ node: neighbor, type });
          } else if (type === 'imports') {
            categories.imports.push({ node: neighbor, type }); // Swapped because direction is reversed
          } else {
            categories.others.push({ node: neighbor, type, direction: 'inbound' });
          }
        }
      }
    });

    return categories;
  }, [selectedNode, links, nodesById]);

  const renderModel = React.useMemo(
    () => buildNodeDetailsRenderModel(selectedNode, detailsState),
    [selectedNode, detailsState]
  );
  const detailSymbols = renderModel.majorBlocks;
  const detailSummary = renderModel.summary || {};
  const displayInbound = detailSummary.inbound_count ?? inbound;
  const displayOutbound = detailSummary.outbound_count ?? outbound;
  const showFileLoading = renderModel.isFile && (detailsState.status === 'idle' || detailsState.status === 'loading');

  const badges = [];
  if (selectedNode?.is_retrieved) badges.push({ label: 'Retrieved', color: '#22d3ee' });
  if (selectedNode?.is_graph_active) badges.push({ label: 'Graph Assist', color: '#4ade80' });

  if (!selectedNode) {
    return (
      <div className={PANEL_CLASS}>
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-sm font-semibold text-text-primary font-mono">Node Details</h3>
            <p className="text-[10px] text-text-muted mt-1 leading-tight">
              Click a node in the graph to inspect its metadata, connections, and retrieval status.
            </p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-text-muted hover:text-text-primary transition-colors p-1 rounded-lg hover:bg-surface-3 self-start shrink-0 ml-2"
              aria-label="Hide panel"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-2">
            <svg className="w-8 h-8 mx-auto text-text-muted/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v4m0 12v4m-10-10h4m12 0h4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
            </svg>
            <p className="text-[10px] text-text-muted font-mono">Select a node</p>
          </div>
        </div>
        <div className="text-[9px] text-text-muted font-mono border-t border-border pt-3">
          Mode: <span className="text-text-primary capitalize">{mode}</span>
        </div>
      </div>
    );
  }

  if (mode === 'retrieval_trace') {
    const stageFlags = selectedNode.stage_flags || {};
    const traceBadges = [
      stageFlags.retrieved && { label: 'Retrieved', color: '#94a3b8' },
      stageFlags.graph_added && { label: 'Graph-added', color: '#a855f7' },
      stageFlags.reranked_in && { label: 'Reranked in', color: '#38bdf8' },
      stageFlags.context_selected && { label: 'Context', color: '#2dd4bf' },
      stageFlags.final && { label: 'Final context', color: '#60a5fa' },
      stageFlags.cited && { label: 'Cited', color: '#34d399' },
      stageFlags.dropped && { label: 'Dropped', color: '#fb7185' },
    ].filter(Boolean);
    const lineRange = traceLineRange(selectedNode);
    const retrieval = selectedNode.retrieval || {};
    const graphMeta = selectedNode.graph || {};
    const traceFlags = selectedNode.flags || {};
    const traceRanks = selectedNode.ranks || {};
    const traceScores = selectedNode.scores || {};
    const traceReasons = selectedNode.reasons || {};
    const isTraceChunk = selectedNode.type === 'chunk';
    const isTraceQuery = selectedNode.type === 'query';
    const isTraceAnswer = selectedNode.type === 'answer';

    return (
      <div className={PANEL_CLASS}>
        <div className="flex justify-between items-start">
          <div>
            <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">
              Retrieval Trace
            </div>
            <h3 className="text-sm font-semibold text-text-primary font-mono mt-0.5 break-words">
              {selectedNode.label || selectedNode.id}
            </h3>
            {selectedNode.path && (
              <p className="text-[10px] text-text-muted font-mono mt-1 break-all">{selectedNode.path}</p>
            )}
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-text-muted hover:text-text-primary transition-colors p-1 rounded-lg hover:bg-surface-3 self-start shrink-0 ml-2"
              aria-label="Hide panel"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {traceBadges.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {traceBadges.map((badge) => (
              <span
                key={badge.label}
                className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] font-mono font-semibold"
                style={{
                  borderColor: `${badge.color}55`,
                  backgroundColor: `${badge.color}18`,
                  color: badge.color,
                }}
              >
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: badge.color }} />
                {badge.label}
              </span>
            ))}
          </div>
        )}

        {isTraceQuery && (
          <TraceSection title="User query">
            <p className="text-[11px] leading-relaxed text-text-primary">{selectedNode.text || selectedNode.label}</p>
            {selectedNode.created_at && (
              <div className="mt-2 text-[9px] text-text-muted font-mono">{selectedNode.created_at}</div>
            )}
            {selectedNode.request && Object.keys(selectedNode.request).length > 0 && (
              <div className="mt-2 grid grid-cols-[92px_1fr] gap-x-2 gap-y-1 text-[9px] font-mono text-text-muted">
                {selectedNode.request.graph_retrieval_mode && <><span>Graph mode</span><span className="text-text-primary">{selectedNode.request.graph_retrieval_mode}</span></>}
                {selectedNode.request.intent && <><span>Intent</span><span className="text-text-primary">{selectedNode.request.intent}</span></>}
                {selectedNode.request.primary_intent && <><span>Primary</span><span className="text-text-primary">{selectedNode.request.primary_intent}</span></>}
              </div>
            )}
          </TraceSection>
        )}

        {isTraceAnswer && (
          <TraceSection title="Assistant answer">
            <p className="text-[11px] leading-relaxed text-text-primary">{selectedNode.text_preview || selectedNode.label}</p>
            <div className="mt-2 grid grid-cols-[72px_1fr] gap-x-2 gap-y-1 text-[9px] font-mono text-text-muted">
              {selectedNode.provider && <><span>Provider</span><span className="text-text-primary">{selectedNode.provider}</span></>}
              {selectedNode.model && <><span>Model</span><span className="text-text-primary break-all">{selectedNode.model}</span></>}
              {selectedNode.trace_version && <><span>Trace</span><span className="text-text-primary">{selectedNode.trace_version}{selectedNode.partial ? ' · partial' : ''}</span></>}
              {selectedNode.created_at && <><span>Created</span><span className="text-text-primary break-all">{selectedNode.created_at}</span></>}
            </div>
            {selectedNode.partial_reason && (
              <p className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-2 py-1.5 text-[9px] leading-relaxed text-amber-100/80">
                {selectedNode.partial_reason}
              </p>
            )}
            {selectedNode.summary && Object.keys(selectedNode.summary).length > 0 && (
              <div className="mt-3 rounded-lg border border-border bg-base/40 p-2">
                <div className="mb-1 text-[9px] uppercase tracking-wider text-text-muted font-bold">Retrieval story</div>
                <p className="text-[10px] leading-relaxed text-text-muted">
                  {(selectedNode.summary.retrieved_count || 0)} chunks were retrieved. Graph Assist added {(selectedNode.summary.graph_added_count || 0)}. {(selectedNode.summary.context_selected_count || selectedNode.summary.final_count || 0)} entered context and {(selectedNode.summary.cited_count || 0)} were cited.
                </p>
              </div>
            )}
          </TraceSection>
        )}

        {isTraceChunk && (
          <>
            <TraceSection title="Evidence chunk">
              <div className="space-y-1.5 text-[10px]">
                {[
                  { label: 'Path', value: selectedNode.path, mono: true },
                  { label: 'Symbol', value: selectedNode.symbol_name },
                  { label: 'Kind', value: selectedNode.kind },
                  { label: 'Lines', value: lineRange },
                ].filter((row) => row.value).map((row) => (
                  <div key={row.label}>
                    <div className="text-[9px] uppercase tracking-wider text-text-muted">{row.label}</div>
                    <div className={`text-text-primary break-all ${row.mono ? 'font-mono' : ''}`}>{row.value}</div>
                  </div>
                ))}
              </div>
              {selectedNode.description && (
                <p className="mt-2 text-[10px] leading-relaxed text-text-muted">{selectedNode.description}</p>
              )}
            </TraceSection>

            <TraceSection title="Trace journey">
              <div className="space-y-1.5 text-[10px] text-text-muted">
                <TraceJourneyRow
                  label="Retrieved"
                  value={traceFlags.retrieved ? `yes${traceRanks.retrieved ? `, rank #${traceRanks.retrieved}` : ''}${traceScores.retrieved !== null && traceScores.retrieved !== undefined ? `, score ${formatTraceNumber(traceScores.retrieved)}` : ''}` : 'no'}
                  active={traceFlags.retrieved}
                />
                <TraceJourneyRow
                  label="Graph Assist"
                  value={traceFlags.graphAdded ? `yes${traceScores.graph !== null && traceScores.graph !== undefined ? `, score ${formatTraceNumber(traceScores.graph)}` : ''}` : 'no'}
                  active={traceFlags.graphAdded}
                />
                <TraceJourneyRow
                  label="Rerank"
                  value={traceFlags.rerankedIn ? `survived${traceRanks.reranked ? `, rank #${traceRanks.reranked}` : ''}` : (traceFlags.dropped ? 'dropped' : 'not recorded')}
                  active={traceFlags.rerankedIn}
                  warning={traceFlags.dropped}
                />
                <TraceJourneyRow
                  label="Context"
                  value={traceFlags.contextSelected ? `selected${traceRanks.context_order ? `, order #${traceRanks.context_order}` : ''}` : 'not selected'}
                  active={traceFlags.contextSelected}
                />
                <TraceJourneyRow
                  label="Final source"
                  value={traceFlags.finalSource ? `shown${traceRanks.display ? `, display #${traceRanks.display}` : ''}` : 'not shown'}
                  active={traceFlags.finalSource}
                />
                <TraceJourneyRow
                  label="Citation"
                  value={traceFlags.cited ? `cited${traceRanks.citation ? `, citation #${traceRanks.citation}` : ''}` : 'not cited'}
                  active={traceFlags.cited}
                />
              </div>
              {selectedNode.explanation && (
                <p className="mt-2 rounded-lg border border-border bg-base/40 px-2 py-1.5 text-[9px] leading-relaxed text-text-muted">
                  {selectedNode.explanation}
                </p>
              )}
              {traceReasons.drop_reason && (
                <p className="mt-2 text-[9px] text-rose-200">Drop reason: {traceReasons.drop_reason}</p>
              )}
            </TraceSection>

            {(retrieval.rank || retrieval.score !== null || retrieval.source) && (
              <TraceSection title="Retrieval">
                <div className="grid grid-cols-[72px_1fr] gap-x-2 gap-y-1 text-[9px] font-mono text-text-muted">
                  {retrieval.rank && <><span>Rank</span><span className="text-text-primary">{retrieval.rank}</span></>}
                  {retrieval.score !== null && retrieval.score !== undefined && <><span>Score</span><span className="text-text-primary">{formatTraceNumber(retrieval.score)}</span></>}
                  {retrieval.source && <><span>Source</span><span className="text-text-primary">{retrieval.source}</span></>}
                </div>
              </TraceSection>
            )}

            {selectedNode.graph && (
              <TraceSection title="Graph Assist">
                <div className="grid grid-cols-[92px_1fr] gap-x-2 gap-y-1 text-[9px] font-mono text-text-muted">
                  {graphMeta.graph_candidate_score !== null && graphMeta.graph_candidate_score !== undefined && (
                    <><span>Score</span><span className="text-text-primary">{formatTraceNumber(graphMeta.graph_candidate_score)}</span></>
                  )}
                  {graphMeta.graph_edge_type && <><span>Edge</span><span className="text-text-primary">{graphMeta.graph_edge_type}</span></>}
                  {graphMeta.graph_anchor_path && <><span>Anchor</span><span className="text-text-primary break-all">{graphMeta.graph_anchor_path}</span></>}
                  {graphMeta.graph_confidence_tier && <><span>Confidence</span><span className="text-text-primary">{graphMeta.graph_confidence_tier}</span></>}
                </div>
                {Array.isArray(graphMeta.graph_score_reasons) && graphMeta.graph_score_reasons.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {graphMeta.graph_score_reasons.slice(0, 8).map((reason) => (
                      <span key={reason} className="rounded-full border border-purple-500/25 bg-purple-500/10 px-2 py-0.5 text-[8px] text-purple-200">
                        {reason}
                      </span>
                    ))}
                  </div>
                )}
              </TraceSection>
            )}

            <div className="flex flex-col gap-1.5 mt-auto pt-3 border-t border-border">
              <button
                onClick={() => selectedNode.chunk_id && onCodeBlockSelect?.({
                  chunk_id: selectedNode.chunk_id,
                  displayName: selectedNode.symbol_name || selectedNode.label,
                  name: selectedNode.symbol_name || selectedNode.label,
                  path: selectedNode.path,
                  start_line: selectedNode.start_line,
                  end_line: selectedNode.end_line,
                  description: selectedNode.description,
                  kind: selectedNode.kind,
                })}
                disabled={!selectedNode.chunk_id}
                className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left disabled:cursor-not-allowed disabled:opacity-50"
              >
                {selectedNode.chunk_id ? 'Preview code block' : 'Code preview unavailable'}
              </button>
              <button
                onClick={() => onFocusNeighbors(selectedNode.id)}
                className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
              >
                Focus trace neighbors
              </button>
              <button
                onClick={onResetFocus}
                className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
              >
                Reset focus
              </button>
            </div>
          </>
        )}

        {!isTraceChunk && (
          <div className="flex flex-col gap-1.5 mt-auto pt-3 border-t border-border">
            <button
              onClick={() => onFocusNeighbors(selectedNode.id)}
              className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
            >
              Focus trace neighbors
            </button>
            <button
              onClick={onResetFocus}
              className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
            >
              Reset focus
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={PANEL_CLASS}>
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">
            {renderModel.isFile ? 'File Details' : 'Selected Node'}
          </div>
          <h3 className="text-sm font-semibold text-text-primary font-mono mt-0.5 break-all">
            {renderModel.title}
          </h3>
          {renderModel.path && renderModel.path !== renderModel.title && (
            <p className="text-[10px] text-text-muted font-mono mt-1 break-all">{renderModel.path}</p>
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors p-1 rounded-lg hover:bg-surface-3 self-start shrink-0 ml-2"
            aria-label="Hide panel"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-[10px] leading-relaxed text-text-primary">
        <div className="font-mono font-semibold text-cyan-200">Selection status</div>
        <div className="mt-1 grid grid-cols-[72px_1fr] gap-x-2 gap-y-0.5 font-mono text-[9px] text-text-muted">
          <span>ID</span>
          <span className="break-all text-text-primary">{selectedNodeId}</span>
          <span>Type</span>
          <span className="text-text-primary">{renderModel.type || 'unknown'}</span>
          <span>Details</span>
          <span className="text-text-primary">{detailsState.status}</span>
          <span>Symbols</span>
          <span className="text-text-primary">{detailSymbols.length}</span>
        </div>
      </div>

      {/* Badges */}
      {badges.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {badges.map((b) => (
            <span
              key={b.label}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-mono font-semibold border"
              style={{
                borderColor: `${b.color}40`,
                backgroundColor: `${b.color}15`,
                color: b.color,
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: b.color, boxShadow: `0 0 4px ${b.color}` }} />
              {b.label}
            </span>
          ))}
        </div>
      )}

      {/* Metadata */}
      <div className="space-y-1.5">
        {[
          { label: 'Type', value: renderModel.type },
          { label: 'Path', value: renderModel.path, mono: true },
          { label: 'Symbol', value: renderModel.symbolName },
          { label: 'Importance', value: renderModel.importance?.toFixed(2) },
        ]
          .filter((r) => r.value)
          .map((r) => (
            <div key={r.label} className="flex flex-col">
              <span className="text-[9px] text-text-muted uppercase tracking-wider">{r.label}</span>
              <span className={`text-[10px] text-text-primary break-all ${r.mono ? 'font-mono' : ''}`}>
                {r.value}
              </span>
            </div>
          ))}
      </div>

      {/* Degree */}
      <div className="grid grid-cols-2 gap-1.5">
        <div className="bg-surface-3 rounded-lg px-2 py-1.5 border border-border">
          <div className="text-[9px] text-text-muted">Inbound</div>
          <div className="text-sm font-semibold text-text-primary font-mono">{displayInbound}</div>
        </div>
        <div className="bg-surface-3 rounded-lg px-2 py-1.5 border border-border">
          <div className="text-[9px] text-text-muted">Outbound</div>
          <div className="text-sm font-semibold text-text-primary font-mono">{displayOutbound}</div>
        </div>
      </div>

      {/* File-level chunk intelligence */}
      {renderModel.isFile && (
        <div className="flex flex-col gap-2.5 border-t border-border pt-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">
              Major code blocks
            </div>
            {detailsState.status === 'ready' && (
              <span className="text-[9px] text-text-muted font-mono">
                {detailSymbols.length}/{detailSummary.symbol_count ?? detailSymbols.length}
              </span>
            )}
          </div>

          {showFileLoading && (
            <div className="rounded-lg border border-border bg-surface-3/70 px-3 py-2 text-[10px] text-text-muted font-mono">
              Loading code blocks...
            </div>
          )}

          {detailsState.status === 'error' && (
            <div className="rounded-lg border border-offline/30 bg-offline/10 px-3 py-2 text-[10px] text-offline leading-relaxed">
              {detailsState.error || 'Unable to load file details.'}
            </div>
          )}

          {detailsState.status === 'empty' && (
            <div className="rounded-lg border border-border bg-surface-3/70 px-3 py-2 text-[10px] text-text-muted leading-relaxed">
              Details unavailable for this file.
            </div>
          )}

          {detailsState.status === 'ready' && detailSymbols.length === 0 && (
            <div className="rounded-lg border border-border bg-surface-3/70 px-3 py-2 text-[10px] text-text-muted leading-relaxed">
              {renderModel.message || 'No major code blocks found for this file.'}
            </div>
          )}

          {detailsState.status === 'ready' && detailSymbols.length > 0 && (
            <div className="flex flex-col gap-2">
              {detailSymbols.map((symbol) => (
                <button
                  type="button"
                  key={`${symbol.chunk_id || symbol.displayName}-${symbol.start_line || 'line'}`}
                  onClick={() => symbol.chunk_id && onCodeBlockSelect?.({ ...symbol, path: renderModel.path })}
                  disabled={!symbol.chunk_id}
                  className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                    symbol.chunk_id
                      ? 'cursor-pointer border-border bg-surface-3/70 hover:border-cyan-500/50 hover:bg-surface-3'
                      : 'cursor-not-allowed border-border bg-surface-3/40 opacity-70'
                  }`}
                  title={symbol.chunk_id ? 'Preview code block' : 'Code preview unavailable'}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-[11px] text-text-primary font-mono font-semibold truncate" title={symbol.displayName}>
                        {symbol.displayName}
                      </div>
                      {symbol.lineLabel && (
                        <div className="text-[9px] text-text-muted font-mono mt-0.5">{symbol.lineLabel}</div>
                      )}
                    </div>
                    <span className="shrink-0 rounded-full border border-border bg-base/70 px-2 py-0.5 text-[8px] uppercase tracking-wide text-text-muted">
                      {symbol.kindLabel}
                    </span>
                  </div>
                  <p className="mt-1.5 text-[10px] leading-relaxed text-text-muted">
                    {symbol.displayDescription}
                  </p>
                  {!symbol.chunk_id && (
                    <div className="mt-2 text-[9px] text-text-muted font-mono">Code preview unavailable</div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {!renderModel.isFile && (
        <div className="rounded-lg border border-border bg-surface-3/70 px-3 py-2 text-[10px] text-text-muted leading-relaxed">
          Detailed chunk descriptions are available for file nodes.
        </div>
      )}

      {/* Connected nodes (Categorized Relationships) */}
      <div className="flex flex-col gap-2.5 mt-1 select-none max-h-56 overflow-y-auto pr-1">
        <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">
          Relationships ({neighbors.size})
        </div>

        {categorizedNeighbors.parents.length > 0 && (
          <div className="space-y-1">
            <div className="text-[8px] uppercase tracking-wider text-text-muted/70 font-bold">
              Parent Node(s)
            </div>
            <div className="flex flex-col gap-0.5">
              {categorizedNeighbors.parents.map((item) => (
                <div key={item.node.id} className="flex items-center gap-1.5 py-0.5 select-text">
                  <span className="w-1 h-3 rounded" style={{ backgroundColor: NODE_COLORS[item.node.type] || '#a0a0a0' }} />
                  <span className="text-[10px] text-text-primary truncate font-mono flex-1" title={item.node.path || item.node.label}>
                    {item.node.label}
                  </span>
                  <span className="text-[8px] text-text-muted/60 font-sans italic">
                    ({item.type})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {categorizedNeighbors.children.length > 0 && (
          <div className="space-y-1">
            <div className="text-[8px] uppercase tracking-wider text-text-muted/70 font-bold">
              Child Node(s)
            </div>
            <div className="flex flex-col gap-0.5">
              {categorizedNeighbors.children.map((item) => (
                <div key={item.node.id} className="flex items-center gap-1.5 py-0.5 select-text">
                  <span className="w-1 h-3 rounded" style={{ backgroundColor: NODE_COLORS[item.node.type] || '#a0a0a0' }} />
                  <span className="text-[10px] text-text-primary truncate font-mono flex-1" title={item.node.path || item.node.label}>
                    {item.node.label}
                  </span>
                  <span className="text-[8px] text-text-muted/60 font-sans italic">
                    ({item.type})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {categorizedNeighbors.imports.length > 0 && (
          <div className="space-y-1">
            <div className="text-[8px] uppercase tracking-wider text-text-muted/70 font-bold">
              Dependencies (Imports)
            </div>
            <div className="flex flex-col gap-0.5">
              {categorizedNeighbors.imports.map((item) => (
                <div key={item.node.id} className="flex items-center gap-1.5 py-0.5 select-text">
                  <span className="w-1 h-3 rounded" style={{ backgroundColor: NODE_COLORS[item.node.type] || '#a0a0a0' }} />
                  <span className="text-[10px] text-text-primary truncate font-mono flex-1" title={item.node.path || item.node.label}>
                    {item.node.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {categorizedNeighbors.importedBy.length > 0 && (
          <div className="space-y-1">
            <div className="text-[8px] uppercase tracking-wider text-text-muted/70 font-bold">
              Required By (Imported By)
            </div>
            <div className="flex flex-col gap-0.5">
              {categorizedNeighbors.importedBy.map((item) => (
                <div key={item.node.id} className="flex items-center gap-1.5 py-0.5 select-text">
                  <span className="w-1 h-3 rounded" style={{ backgroundColor: NODE_COLORS[item.node.type] || '#a0a0a0' }} />
                  <span className="text-[10px] text-text-primary truncate font-mono flex-1" title={item.node.path || item.node.label}>
                    {item.node.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {categorizedNeighbors.others.length > 0 && (
          <div className="space-y-1">
            <div className="text-[8px] uppercase tracking-wider text-text-muted/70 font-bold">
              Other Relations
            </div>
            <div className="flex flex-col gap-0.5">
              {categorizedNeighbors.others.map((item) => (
                <div key={item.node.id} className="flex items-center gap-1.5 py-0.5 select-text">
                  <span className="w-1 h-3 rounded" style={{ backgroundColor: NODE_COLORS[item.node.type] || '#a0a0a0' }} />
                  <span className="text-[10px] text-text-primary truncate font-mono flex-1" title={item.node.path || item.node.label}>
                    {item.node.label}
                  </span>
                  <span className="text-[8px] text-text-muted/60 font-sans italic">
                    ({item.type} {item.direction})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-1.5 mt-auto pt-3 border-t border-border">
        <button
          onClick={() => onFocusNeighbors(selectedNode.id)}
          className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
        >
          Focus neighbors
        </button>
        <button
          onClick={() => onAskAbout(selectedNode)}
          className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
          title="Pre-fill chat input with a question about this node"
        >
          Ask CodeSeek about this
        </button>
        <button
          onClick={onResetFocus}
          className="px-2.5 py-1.5 rounded-lg bg-surface-3 border border-border text-[10px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors text-left"
        >
          Reset focus
        </button>
      </div>
    </div>
  );
}

function TraceSection({ title, children }) {
  return (
    <div className="rounded-lg border border-border bg-surface-3/70 px-3 py-2">
      <div className="mb-1.5 text-[9px] uppercase tracking-wider text-text-muted font-bold">{title}</div>
      {children}
    </div>
  );
}

function TraceJourneyRow({ label, value, active = false, warning = false }) {
  const dotColor = warning ? '#fb7185' : active ? '#34d399' : '#64748b';
  return (
    <div className="grid grid-cols-[88px_1fr] gap-2 items-start">
      <div className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider text-text-muted">
        <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: dotColor }} />
        {label}
      </div>
      <div className={`${warning ? 'text-rose-200' : active ? 'text-text-primary' : 'text-text-muted'} font-mono text-[9px]`}>
        {value}
      </div>
    </div>
  );
}

function traceLineRange(node) {
  const start = node?.start_line;
  const end = node?.end_line;
  if (start && end && start !== end) return `${start}-${end}`;
  if (start) return String(start);
  return '';
}

function formatTraceNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return number % 1 === 0 ? String(number) : number.toFixed(3);
}
