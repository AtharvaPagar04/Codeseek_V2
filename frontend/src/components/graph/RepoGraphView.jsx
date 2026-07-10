import React, { useRef, useCallback, useEffect, useState } from 'react';
import { useRepoGraph } from '../../hooks/useRepoGraph';
import RepoGraphSidebar from './RepoGraphSidebar';
import RepoGraphCanvas from './RepoGraphCanvas';
import RepoGraphDetailsPanel from './RepoGraphDetailsPanel';
import GraphCodePreviewModal from './GraphCodePreviewModal';

/**
 * Main repo graph visualization view — three-panel layout with
 * sidebar controls, interactive canvas, and details panel.
 */
export default function RepoGraphView({
  sessionId,
  onAskAbout,
  requestedMode = null,
  traceSourceMode = 'latest',
  selectedTraceMessageId = null,
  onTraceSourceChange,
}) {
  const graph = useRepoGraph(sessionId, {
    enabled: true,
    modeOverride: requestedMode,
    traceSourceMode,
    traceMessageId: selectedTraceMessageId,
  });
  const fitViewRef = useRef(null);
  const [activeCodeBlock, setActiveCodeBlock] = useState(null);
  const [showDetails, setShowDetails] = useState(true);
  const selectedNode = graph.selectedNode;
  const selectedNodeId = graph.selectedNodeId;

  const handleFitView = useCallback(() => {
    if (fitViewRef.current) fitViewRef.current();
  }, []);

  const handleFocusNeighbors = useCallback(
    (nodeId) => {
      graph.setFocusNodeId(nodeId);
    },
    [graph]
  );

  const handleResetFocus = useCallback(() => {
    setActiveCodeBlock(null);
    graph.setFocusNodeId(null);
    graph.clearSelection();
  }, [graph]);

  const handleResetView = useCallback(() => {
    setActiveCodeBlock(null);
    graph.resetView();
  }, [graph]);

  const handleBackgroundClick = useCallback(() => {
    setActiveCodeBlock(null);
    graph.clearSelection();
  }, [graph]);

  const handleNodeSelect = useCallback(
    (node) => {
      graph.setSelectedNodeId(node?.id || null);
      if (node) {
        setShowDetails(true);
      }
    },
    [graph]
  );

  const handleAskAbout = useCallback(
    (node) => {
      if (onAskAbout) {
        const query = node.path
          ? `Show me the code for ${node.path}`
          : node.symbol_name
            ? `What does ${node.symbol_name} do?`
            : `Tell me about ${node.label}`;
        onAskAbout(query);
      }
    },
    [onAskAbout]
  );

  // Clear selection when the session changes — intentionally omitting `graph`
  // from deps because the hook returns a new object ref every render, which
  // would cause this effect to fire continuously and wipe out selections.
  useEffect(() => {
    setActiveCodeBlock(null);
    graph.clearSelection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Ensure details panel is visible when mode changes (e.g. entering trace mode)
  useEffect(() => {
    setShowDetails(true);
  }, [graph.mode]);

  const lastRawDataRef = useRef(null);

  // Auto-select answer/query node when a new retrieval trace is loaded
  useEffect(() => {
    if (graph.mode !== 'retrieval_trace') {
      lastRawDataRef.current = null;
      return;
    }
    if (graph.rawData && graph.rawData !== lastRawDataRef.current) {
      lastRawDataRef.current = graph.rawData;
      const answerNode = graph.graphData.nodes.find((n) => n.type === 'answer');
      const queryNode = graph.graphData.nodes.find((n) => n.type === 'query');
      const defaultNode = answerNode || queryNode;
      if (defaultNode) {
        graph.setSelectedNodeId(defaultNode.id);
      }
    }
  }, [graph.mode, graph.rawData, graph.graphData.nodes, graph.setSelectedNodeId]);

  // Endpoint missing state
  if (graph.endpointMissing) {
    return (
      <div className="flex-1 flex items-center justify-center bg-base">
        <div className="text-center space-y-3 max-w-sm px-6">
          <div className="w-12 h-12 mx-auto rounded-2xl bg-surface-3 border border-border flex items-center justify-center">
            <svg className="w-6 h-6 text-text-muted/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-sm font-semibold text-text-primary font-mono">Graph Endpoint Not Available</h3>
          <p className="text-[11px] text-text-muted leading-relaxed">
            The graph visualization backend endpoint is not available yet.
            This feature requires <code className="px-1 py-0.5 bg-surface-3 rounded text-text-primary text-[10px]">GET /api/v1/sessions/:id/graph</code> to be implemented.
          </p>
          <button
            onClick={graph.retry}
            className="px-4 py-2 rounded-xl bg-surface-3 border border-border text-[11px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Error state
  if (graph.error && !graph.loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-base">
        <div className="text-center space-y-3 max-w-sm px-6">
          <div className="w-12 h-12 mx-auto rounded-2xl bg-offline/10 border border-offline/20 flex items-center justify-center">
            <svg className="w-6 h-6 text-offline/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h3 className="text-sm font-semibold text-text-primary font-mono">Failed to Load Graph</h3>
          <p className="text-[11px] text-text-muted leading-relaxed">{graph.error}</p>
          <button
            onClick={graph.retry}
            className="px-4 py-2 rounded-xl bg-surface-3 border border-border text-[11px] font-mono text-text-muted hover:text-text-primary hover:border-text-muted transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Loading state
  if (graph.loading && !graph.rawData) {
    return (
      <div className="flex-1 flex items-center justify-center bg-base">
        <div className="text-center space-y-3">
          <div className="flex items-center justify-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-dot-1" />
            <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-dot-2" />
            <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-dot-3" />
          </div>
          <p className="text-[11px] text-text-muted font-mono">Loading graph data…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 overflow-hidden">
      <RepoGraphSidebar
        mode={graph.mode}
        modes={graph.modes}
        onModeChange={graph.setMode}
        traceSourceMode={traceSourceMode}
        selectedTraceMessageId={selectedTraceMessageId}
        traceMessages={graph.traceMessages}
        traceMessagesStatus={graph.traceMessagesStatus}
        traceMessagesError={graph.traceMessagesError}
        onTraceSourceChange={onTraceSourceChange}
        searchQuery={graph.searchQuery}
        onSearchChange={graph.setSearchQuery}
        nodeTypeFilter={graph.nodeTypeFilter}
        onNodeTypeFilterChange={graph.setNodeTypeFilter}
        edgeTypeFilter={graph.edgeTypeFilter}
        onEdgeTypeFilterChange={graph.setEdgeTypeFilter}
        summary={graph.graphData.summary}
        onResetView={handleResetView}
        onFitView={handleFitView}
      />

      <RepoGraphCanvas
        graphData={graph.graphData}
        selectedNode={selectedNode}
        selectedNodeId={selectedNodeId}
        hoveredNodeId={graph.hoveredNodeId}
        onNodeClick={graph.setSelectedNodeId}
        onNodeSelect={handleNodeSelect}
        onNodeHover={graph.setHoveredNodeId}
        onBackgroundClick={handleBackgroundClick}
        fitViewRef={fitViewRef}
      />

      {showDetails && (
        <RepoGraphDetailsPanel
          sessionId={sessionId}
          selectedNode={selectedNode}
          nodes={graph.graphData.nodes}
          links={graph.graphData.links}
          mode={graph.mode}
          onFocusNeighbors={handleFocusNeighbors}
          onResetFocus={handleResetFocus}
          onAskAbout={handleAskAbout}
          onCodeBlockSelect={setActiveCodeBlock}
          onClose={() => setShowDetails(false)}
        />
      )}

      <GraphCodePreviewModal
        sessionId={sessionId}
        codeBlock={activeCodeBlock}
        onClose={() => setActiveCodeBlock(null)}
      />
    </div>
  );
}
