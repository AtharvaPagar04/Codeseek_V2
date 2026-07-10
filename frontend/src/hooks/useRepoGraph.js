import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  fetchLatestRetrievalTrace,
  fetchRetrievalTrace,
  fetchRetrievalTraceMessages,
  fetchSessionGraph,
} from '../utils/api';
import { transformGraphData } from '../components/graph/graphTransform';
import { transformRetrievalTraceData } from '../components/graph/retrievalTraceTransform';

const MODES = ['imports', 'structure', 'retrieval_trace'];

/**
 * Hook for managing repo graph state — fetching, filtering, selection.
 */
export function useRepoGraph(
  sessionId,
  {
    enabled = false,
    modeOverride = null,
    traceSourceMode = 'latest',
    traceMessageId = null,
  } = {}
) {
  const [rawData, setRawData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [endpointMissing, setEndpointMissing] = useState(false);
  const [traceMessages, setTraceMessages] = useState([]);
  const [traceMessagesStatus, setTraceMessagesStatus] = useState('idle');
  const [traceMessagesError, setTraceMessagesError] = useState(null);

  // View controls
  const [mode, setMode] = useState(() => (MODES.includes(modeOverride) ? modeOverride : 'imports'));
  const [searchQuery, setSearchQuery] = useState('');
  const [nodeTypeFilter, setNodeTypeFilter] = useState([]);
  const [edgeTypeFilter, setEdgeTypeFilter] = useState([]);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [focusNodeId, setFocusNodeId] = useState(null);

  const fetchIdRef = useRef(0);
  const traceMessagesFetchIdRef = useRef(0);
  const effectiveFocusNodeId = mode === 'retrieval_trace' ? '' : focusNodeId;

  const fetchGraph = useCallback(async () => {
    if (!sessionId || !enabled) return;
    const id = ++fetchIdRef.current;
    setLoading(true);
    setError(null);
    setEndpointMissing(false);

    try {
      const data = mode === 'retrieval_trace'
        ? (traceSourceMode === 'message' && traceMessageId
            ? await fetchRetrievalTrace(sessionId, traceMessageId)
            : await fetchLatestRetrievalTrace(sessionId))
        : await fetchSessionGraph(sessionId, {
            mode,
            limitNodes: 250,
            limitEdges: 500,
            depth: 2,
            focusNodeId: effectiveFocusNodeId || undefined,
          });

      if (id !== fetchIdRef.current) return; // stale

      if (data === null) {
        setEndpointMissing(true);
        setRawData(null);
      } else {
        setRawData(data);
      }
    } catch (err) {
      if (id !== fetchIdRef.current) return;
      setError(err.message || 'Failed to load graph');
    } finally {
      if (id === fetchIdRef.current) setLoading(false);
    }
  }, [sessionId, enabled, mode, effectiveFocusNodeId, traceSourceMode, traceMessageId]);

  const fetchTraceMessages = useCallback(async () => {
    if (!sessionId || !enabled || mode !== 'retrieval_trace') return;
    const id = ++traceMessagesFetchIdRef.current;
    setTraceMessagesStatus('loading');
    setTraceMessagesError(null);
    try {
      const data = await fetchRetrievalTraceMessages(sessionId, { limit: 30 });
      if (id !== traceMessagesFetchIdRef.current) return;
      setTraceMessages(Array.isArray(data?.items) ? data.items : []);
      setTraceMessagesStatus('ready');
    } catch (err) {
      if (id !== traceMessagesFetchIdRef.current) return;
      setTraceMessages([]);
      setTraceMessagesError(err.message || 'Failed to load trace messages');
      setTraceMessagesStatus('error');
    }
  }, [sessionId, enabled, mode]);

  // Fetch when mode/session/enabled changes
  useEffect(() => {
    if (enabled) {
      fetchGraph();
    } else {
      setRawData(null);
      setError(null);
      setEndpointMissing(false);
    }
  }, [fetchGraph, enabled]);

  useEffect(() => {
    if (enabled && mode === 'retrieval_trace') {
      fetchTraceMessages();
    } else {
      setTraceMessages([]);
      setTraceMessagesStatus('idle');
      setTraceMessagesError(null);
    }
  }, [fetchTraceMessages, enabled, mode]);

  // Reset selection when session changes
  useEffect(() => {
    setSelectedNodeId(null);
    setHoveredNodeId(null);
    setFocusNodeId(null);
    setSearchQuery('');
    setNodeTypeFilter([]);
    setEdgeTypeFilter([]);
    setTraceMessages([]);
    setTraceMessagesStatus('idle');
    setTraceMessagesError(null);
    setMode(MODES.includes(modeOverride) ? modeOverride : 'imports');
  }, [sessionId, modeOverride]);

  useEffect(() => {
    if (MODES.includes(modeOverride)) {
      setMode(modeOverride);
    }
  }, [modeOverride]);

  // Mode-specific node/edge types differ, so avoid carrying stale filters across modes.
  useEffect(() => {
    setSelectedNodeId(null);
    setHoveredNodeId(null);
    setFocusNodeId(null);
    setNodeTypeFilter([]);
    setEdgeTypeFilter([]);
  }, [mode]);

  // Transformed data (memoized, recomputed only on raw data or filter changes)
  const graphData = useMemo(() => {
    if (!rawData) return { nodes: [], links: [], summary: { node_count: 0, edge_count: 0, node_types: {}, edge_types: {} } };
    if (mode === 'retrieval_trace') {
      return transformRetrievalTraceData(rawData, {
        nodeTypeFilter,
        edgeTypeFilter,
        searchQuery,
      });
    }
    return transformGraphData(rawData, {
      nodeTypeFilter,
      edgeTypeFilter,
      searchQuery,
    });
  }, [rawData, mode, nodeTypeFilter, edgeTypeFilter, searchQuery]);

  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return graphData.nodes.find((n) => n.id === selectedNodeId) || null;
  }, [selectedNodeId, graphData.nodes]);

  return {
    // Data
    graphData,
    rawData,
    traceMessages,
    traceMessagesStatus,
    traceMessagesError,
    selectedNode,
    loading,
    error,
    endpointMissing,

    // View state
    mode,
    modes: MODES,
    searchQuery,
    nodeTypeFilter,
    edgeTypeFilter,
    selectedNodeId,
    hoveredNodeId,
    focusNodeId,

    // Actions
    setMode,
    setSearchQuery,
    setNodeTypeFilter,
    setEdgeTypeFilter,
    setSelectedNodeId,
    setHoveredNodeId,
    setFocusNodeId,
    retry: fetchGraph,
    retryTraceMessages: fetchTraceMessages,
    clearSelection: () => {
      setSelectedNodeId(null);
      setHoveredNodeId(null);
    },
    resetView: () => {
      setSelectedNodeId(null);
      setHoveredNodeId(null);
      setFocusNodeId(null);
      setSearchQuery('');
      setNodeTypeFilter([]);
      setEdgeTypeFilter([]);
    },
  };
}
