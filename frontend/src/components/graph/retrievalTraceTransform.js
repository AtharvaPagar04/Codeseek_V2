const DEFAULT_MAX_NODES = 250;
const DEFAULT_MAX_EDGES = 500;

const STAGE_TO_EDGE_TYPE = {
  retrieved: 'retrieved',
  graph_added: 'graph_added',
  reranked_in: 'reranked',
  context_selected: 'context_selected',
  final_source: 'final_context',
  final: 'final_context',
  cited: 'cited',
  dropped: 'dropped',
};

export function transformRetrievalTraceData(response, options = {}) {
  if (!response || (!Array.isArray(response.nodes) && !Array.isArray(response.chunks))) {
    return emptyTraceGraph();
  }
  const hasResponseNodes = Array.isArray(response.nodes) && response.nodes.length > 0;
  const hasResponseChunks = Array.isArray(response.chunks) && response.chunks.length > 0;
  if (!hasResponseNodes && !hasResponseChunks) {
    return emptyTraceGraph(response);
  }

  const {
    nodeTypeFilter = [],
    edgeTypeFilter = [],
    searchQuery = '',
    maxNodes = DEFAULT_MAX_NODES,
    maxEdges = DEFAULT_MAX_EDGES,
  } = options;

  const responseNodes = hasResponseNodes
    ? response.nodes
    : buildNodesFromV2Chunks(response);
  // Normalize then deduplicate by ID (last occurrence wins for V1 backend duplicates)
  const normalizedAll = responseNodes.map((node) => normalizeTraceNode(enrichTraceNode(node, response))).filter(Boolean);

  // Unify different ID representations of the same chunk
  const pathSymbolStartToId = new Map();
  for (const node of normalizedAll) {
    if (node.type === 'chunk') {
      const path = node.path || '';
      const symbol = node.symbol_name || '';
      const start = node.start_line || 0;
      if (path && !node.id.startsWith('chunk:path:')) {
        pathSymbolStartToId.set(`${path}::${symbol}::${start}`, node.id);
        if (symbol) {
          pathSymbolStartToId.set(`${path}::::${start}`, node.id);
        }
      }
    }
  }

  const idTranslation = new Map();
  for (const node of normalizedAll) {
    if (node.type === 'chunk') {
      const path = node.path || '';
      const symbol = node.symbol_name || '';
      const start = node.start_line || 0;
      const key = `${path}::${symbol}::${start}`;
      const fallbackKey = `${path}::::${start}`;
      const unifiedId = pathSymbolStartToId.get(key) || pathSymbolStartToId.get(fallbackKey);
      if (unifiedId && node.id !== unifiedId) {
        idTranslation.set(node.id, unifiedId);
        node.id = unifiedId;
      }
    }
  }

  const nodeById = new Map();
  for (const node of normalizedAll) {
    const existing = nodeById.get(node.id);
    if (existing) {
      existing.stage_flags = {
        ...existing.stage_flags,
        ...node.stage_flags,
      };
      if (node.trace_flags) {
        existing.trace_flags = Array.from(new Set([...(existing.trace_flags || []), ...node.trace_flags]));
      }
      const reconciledDropped = existing.stage_flags.dropped && 
                                !existing.stage_flags.cited && 
                                !existing.stage_flags.final && 
                                !existing.stage_flags.context_selected;
      existing.stage_flags.dropped = reconciledDropped;
      existing.is_trace_dropped = reconciledDropped;
      existing.is_retrieved = existing.is_retrieved || node.is_retrieved;
      existing.is_graph_active = existing.is_graph_active || node.is_graph_active;
      existing.is_trace_final = existing.is_trace_final || node.is_trace_final;
      existing.is_trace_cited = existing.is_trace_cited || node.is_trace_cited;
    } else {
      nodeById.set(node.id, node);
    }
  }
  let nodes = Array.from(nodeById.values());
  const stageFilters = new Set(edgeTypeFilter.map((type) => String(type).toLowerCase()));

  if (nodeTypeFilter.length > 0) {
    const allowed = new Set(nodeTypeFilter.map((type) => String(type).toLowerCase()));
    nodes = nodes.filter((node) => allowed.has(node.type));
  }

  if (stageFilters.size > 0) {
    nodes = nodes.filter((node) => node.type !== 'chunk' || chunkMatchesStageFilters(node, stageFilters));
  }

  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase();
    nodes = nodes.filter((node) => (
      String(node.label || '').toLowerCase().includes(q)
      || String(node.path || '').toLowerCase().includes(q)
      || String(node.symbol_name || '').toLowerCase().includes(q)
      || String(node.description || '').toLowerCase().includes(q)
      || node.type === 'query'
      || node.type === 'answer'
    ));
  }

  if (nodes.length > maxNodes) {
    nodes.sort((a, b) => b.importance - a.importance);
    nodes = nodes.slice(0, maxNodes);
  }

  const nodeIds = new Set(nodes.map((node) => node.id));
  const responseEdges = Array.isArray(response.edges) && response.edges.length > 0
    ? response.edges
    : buildEdgesFromV2Chunks(response);
  const allLinks = responseEdges.map((edge) => {
    const newSource = idTranslation.get(edge.source) || edge.source;
    const newTarget = idTranslation.get(edge.target) || edge.target;
    return normalizeTraceEdge({
      ...edge,
      source: newSource,
      target: newTarget,
    });
  }).filter((edge) => (
    nodeIds.has(edge.source) && nodeIds.has(edge.target)
  ));
  // Deduplicate: keep only the highest-priority edge per source→target pair
  const linkByPair = new Map();
  for (const link of allLinks) {
    const pairKey = `${link.source}→${link.target}`;
    const existing = linkByPair.get(pairKey);
    if (!existing || (EDGE_PRIORITY[link.type] || 0) > (EDGE_PRIORITY[existing.type] || 0)) {
      linkByPair.set(pairKey, link);
    }
  }
  let links = Array.from(linkByPair.values());

  if (stageFilters.size > 0) {
    links = links.filter((edge) => stageFilters.has(edge.type));
  }

  if (links.length > maxEdges) {
    links = links.slice(0, maxEdges);
  }

  const connectedIds = new Set();
  for (const link of links) {
    connectedIds.add(link.source);
    connectedIds.add(link.target);
  }
  nodes = nodes.filter((node) => node.type === 'chunk' || connectedIds.has(node.id) || links.length === 0);

  const degrees = {};
  for (const link of links) {
    const src = typeof link.source === 'object' ? link.source.id : link.source;
    const tgt = typeof link.target === 'object' ? link.target.id : link.target;
    if (src) degrees[src] = (degrees[src] || 0) + 1;
    if (tgt) degrees[tgt] = (degrees[tgt] || 0) + 1;
  }

  nodes = nodes.map((node) => {
    const deg = degrees[node.id] || 0;
    const base = node.size || sizeForTraceNode(node.type, node.stage_flags);
    const scale = 1.0 + Math.min(Math.log1p(deg) * 0.35, 1.2);
    return {
      ...node,
      size: base * scale,
      degree: deg,
    };
  });

  return {
    meta: {
      trace_version: response.trace_version || 'v1-fallback',
      partial: Boolean(response.partial),
      partial_reason: response.partial_reason || '',
      assistant_message_id: response.assistant_message_id || response.message_id || null,
      user_message_id: response.user_message_id || null,
      request: response.request || {},
    },
    traceVersion: response.trace_version || 'v1-fallback',
    partial: Boolean(response.partial),
    partialReason: response.partial_reason || '',
    assistantMessageId: response.assistant_message_id || response.message_id || null,
    userMessageId: response.user_message_id || null,
    nodes,
    links,
    summary: buildTraceSummary(nodes, links, response),
    stages: response.stages || emptyStages(),
    chunksById: buildChunksById(response.chunks || [], nodes),
    status: response.status || 'empty',
    message: response.message || '',
    query: response.query || null,
    answer: response.answer || null,
  };
}

export function normalizeTraceNode(raw) {
  if (!raw || !raw.id) return null;
  const type = String(raw.type || 'chunk').toLowerCase();
  const rawStageFlags = {
    retrieved: Boolean(raw.stage_flags?.retrieved),
    graph_added: Boolean(raw.stage_flags?.graph_added),
    final: Boolean(raw.stage_flags?.final),
    cited: Boolean(raw.stage_flags?.cited),
    reranked_in: Boolean(raw.stage_flags?.reranked_in),
    context_selected: Boolean(raw.stage_flags?.context_selected),
    final_source: Boolean(raw.stage_flags?.final_source),
    dropped: Boolean(raw.stage_flags?.dropped),
  };
  const traceProvenance = raw.trace_provenance || raw.provenance || {};

  // Reconcile contradictory flags: answer-bound evidence overrides dropped
  const reachedAnswer = rawStageFlags.context_selected || rawStageFlags.final_source
    || rawStageFlags.final || rawStageFlags.cited
    || Boolean(traceProvenance.used_in_context)
    || Boolean(traceProvenance.shown_as_final_source)
    || Boolean(traceProvenance.cited_in_answer);
  const effectiveDropped = (rawStageFlags.dropped || Boolean(traceProvenance.was_dropped)) && !reachedAnswer;

  const stageFlags = {
    ...rawStageFlags,
    dropped: effectiveDropped,
  };

  return {
    id: raw.id,
    label: raw.label || raw.symbol_name || shortPath(raw.path) || raw.id,
    type,
    path: raw.path || null,
    symbol_name: raw.symbol_name || null,
    chunk_id: raw.chunk_id || null,
    kind: raw.kind || null,
    start_line: raw.start_line || null,
    end_line: raw.end_line || null,
    description: raw.description || '',
    stage_flags: stageFlags,
    trace_flags: Array.isArray(raw.trace_flags) ? raw.trace_flags : flagsFromStageFlags(stageFlags),
    trace_provenance: traceProvenance,
    flags: {
      retrieved: stageFlags.retrieved,
      graphAdded: stageFlags.graph_added,
      rerankedIn: stageFlags.reranked_in || Boolean(traceProvenance.survived_rerank),
      contextSelected: stageFlags.context_selected || Boolean(traceProvenance.used_in_context),
      finalSource: stageFlags.final_source || Boolean(traceProvenance.shown_as_final_source),
      cited: stageFlags.cited,
      dropped: effectiveDropped,
    },
    ranks: raw.ranks || {},
    scores: raw.scores || {},
    reasons: raw.reasons || {},
    explanation: raw.explanation || '',
    retrieval: raw.retrieval || null,
    graph: raw.graph || null,
    text: raw.text || '',
    text_preview: raw.text_preview || '',
    trace_version: raw.trace_version || null,
    partial: Boolean(raw.partial),
    partial_reason: raw.partial_reason || '',
    assistant_message_id: raw.assistant_message_id || null,
    user_message_id: raw.user_message_id || null,
    request: raw.request || null,
    model: raw.model || '',
    provider: raw.provider || '',
    created_at: raw.created_at || null,
    summary: raw.summary || {},
    size: raw.size || sizeForTraceNode(type, stageFlags),
    importance: raw.importance ?? importanceForTraceNode(type, stageFlags),
    is_retrieved: stageFlags.retrieved,
    is_graph_active: stageFlags.graph_added,
    is_trace_final: stageFlags.final,
    is_trace_cited: stageFlags.cited,
    is_trace_context: stageFlags.context_selected,
    is_trace_dropped: effectiveDropped,
    is_trace_reranked: stageFlags.reranked_in || Boolean(traceProvenance.survived_rerank),
    metadata: raw.metadata || {},
  };
}

function enrichTraceNode(raw, response) {
  if (!raw || !response) return raw;
  if (raw.type === 'query') {
    return {
      ...raw,
      request: raw.request || response.request || {},
      user_message_id: raw.user_message_id || response.user_message_id || null,
    };
  }
  if (raw.type === 'answer') {
    return {
      ...raw,
      trace_version: raw.trace_version || response.trace_version || null,
      partial: raw.partial ?? Boolean(response.partial),
      partial_reason: raw.partial_reason || response.partial_reason || '',
      assistant_message_id: raw.assistant_message_id || response.assistant_message_id || response.message_id || null,
      summary: raw.summary || response.summary || {},
    };
  }
  return raw;
}

export function normalizeTraceEdge(raw) {
  const type = String(raw?.type || 'retrieved').toLowerCase();
  return {
    id: raw.id || `edge:${raw.source}:${type}:${raw.target}`,
    source: raw.source,
    target: raw.target,
    type,
    label: raw.label || type,
    weight: edgeWeight(type),
    metadata: raw.metadata || {},
  };
}

function chunkMatchesStageFilters(node, stageFilters) {
  const flags = node.stage_flags || {};
  for (const [flag, edgeType] of Object.entries(STAGE_TO_EDGE_TYPE)) {
    if (flags[flag] && stageFilters.has(edgeType)) return true;
  }
  return false;
}

function buildTraceSummary(nodes, links, response) {
  const nodeTypes = {};
  const edgeTypes = {};
  const chunks = nodes.filter((node) => node.type === 'chunk');
  for (const node of nodes) nodeTypes[node.type] = (nodeTypes[node.type] || 0) + 1;
  for (const link of links) edgeTypes[link.type] = (edgeTypes[link.type] || 0) + 1;
  return {
    ...(response.summary || {}),
    node_count: nodes.length,
    edge_count: links.length,
    node_types: nodeTypes,
    edge_types: edgeTypes,
    retrieved_count: chunks.filter((node) => node.stage_flags?.retrieved).length,
    graph_added_count: chunks.filter((node) => node.stage_flags?.graph_added).length,
    reranked_count: response.summary?.reranked_count ?? chunks.filter((node) => node.stage_flags?.reranked_in).length,
    context_selected_count: response.summary?.context_selected_count ?? chunks.filter((node) => node.stage_flags?.context_selected).length,
    final_source_count: response.summary?.final_source_count ?? chunks.filter((node) => node.stage_flags?.final_source).length,
    final_count: chunks.filter((node) => node.stage_flags?.final).length,
    cited_count: chunks.filter((node) => node.stage_flags?.cited).length,
    dropped_count: response.summary?.dropped_count ?? chunks.filter((node) => node.stage_flags?.dropped).length,
    status: response.status || 'empty',
    message: response.message || '',
    partial: Boolean(response.partial),
    partial_reason: response.partial_reason || '',
    trace_version: response.trace_version || 'v1-fallback',
  };
}

function emptyTraceGraph(response = {}) {
  return {
    nodes: [],
    links: [],
    summary: {
      node_count: 0,
      edge_count: 0,
      node_types: {},
      edge_types: {},
      retrieved_count: 0,
      graph_added_count: 0,
      reranked_count: 0,
      context_selected_count: 0,
      final_source_count: 0,
      final_count: 0,
      cited_count: 0,
      dropped_count: 0,
      status: response.status || 'empty',
      message: response.message || '',
      partial: Boolean(response.partial),
      partial_reason: response.partial_reason || '',
      trace_version: response.trace_version || '',
    },
    stages: emptyStages(),
    chunksById: {},
    status: response.status || 'empty',
    message: response.message || '',
    meta: {
      trace_version: response.trace_version || '',
      partial: Boolean(response.partial),
      partial_reason: response.partial_reason || '',
      assistant_message_id: response.assistant_message_id || response.message_id || null,
      user_message_id: response.user_message_id || null,
      request: response.request || {},
    },
    traceVersion: response.trace_version || '',
    partial: Boolean(response.partial),
    partialReason: response.partial_reason || '',
    assistantMessageId: response.assistant_message_id || response.message_id || null,
    userMessageId: response.user_message_id || null,
  };
}

function sizeForTraceNode(type, flags) {
  if (type === 'query' || type === 'answer') return 8;
  if (flags.cited) return 7;
  if (flags.final) return 6;
  if (flags.graph_added) return 6;
  if (flags.dropped) return 4;
  return 5;
}

function importanceForTraceNode(type, flags) {
  if (type === 'query' || type === 'answer') return 1;
  if (flags.cited) return 0.95;
  if (flags.final) return 0.85;
  if (flags.graph_added) return 0.75;
  if (flags.retrieved) return 0.55;
  if (flags.dropped) return 0.3;
  return 0.4;
}

function edgeWeight(type) {
  if (type === 'cited') return 2.4;
  if (type === 'final_context') return 1.8;
  if (type === 'context_selected') return 1.7;
  if (type === 'graph_added') return 1.5;
  if (type === 'dropped') return 0.7;
  return 1;
}

function shortPath(path) {
  if (!path) return '';
  const parts = String(path).split('/');
  return parts[parts.length - 1] || path;
}

function emptyStages() {
  return {
    retrieved_candidates: [],
    graph_added_candidates: [],
    reranked_candidates: [],
    context_selected_candidates: [],
    final_sources: [],
    cited_sources: [],
  };
}

function flagsFromStageFlags(flags) {
  const result = [];
  if (flags.retrieved) result.push('retrieved');
  if (flags.graph_added) result.push('graph_added');
  if (flags.reranked_in) result.push('reranked_in');
  if (flags.context_selected) result.push('context_selected');
  if (flags.final_source) result.push('final_source');
  if (flags.cited) result.push('cited');
  if (flags.dropped) result.push('dropped');
  return result;
}

function buildChunksById(chunks, nodes) {
  const byId = {};
  for (const chunk of chunks || []) {
    if (!chunk) continue;
    const key = chunk.chunk_id || chunk.id || chunk.relative_path;
    if (key) byId[key] = chunk;
  }
  for (const node of nodes || []) {
    if (node.type !== 'chunk') continue;
    const key = node.chunk_id || node.id || node.path;
    if (key && !byId[key]) byId[key] = node;
  }
  return byId;
}

function buildNodesFromV2Chunks(response) {
  const query = response.query || {};
  const answer = response.answer || {};
  const queryId = query.id || `query:${response.user_message_id || 'latest'}`;
  const answerId = answer.id || `answer:${response.assistant_message_id || response.message_id || 'latest'}`;
  return [
    {
      id: queryId,
      type: 'query',
      label: query.text || 'User query',
      text: query.text || '',
      created_at: query.created_at || null,
      request: response.request || {},
      user_message_id: response.user_message_id || null,
    },
    ...(response.chunks || []).map((chunk) => ({
      id: `chunk:${chunk.id || chunk.chunk_id || chunk.relative_path}`,
      type: 'chunk',
      label: chunk.label || chunk.symbol_name || shortPath(chunk.relative_path),
      chunk_id: chunk.chunk_id || '',
      path: chunk.relative_path || chunk.path || '',
      symbol_name: chunk.symbol_name || '',
      kind: chunk.kind || 'chunk',
      start_line: chunk.start_line || null,
      end_line: chunk.end_line || null,
      description: chunk.description || '',
      stage_flags: {
        retrieved: Boolean(chunk.provenance?.was_retrieved),
        graph_added: Boolean(chunk.provenance?.was_graph_added),
        reranked_in: Boolean(chunk.provenance?.survived_rerank),
        context_selected: Boolean(chunk.provenance?.used_in_context),
        final_source: Boolean(chunk.provenance?.shown_as_final_source),
        final: Boolean(chunk.provenance?.used_in_context || chunk.provenance?.shown_as_final_source),
        cited: Boolean(chunk.provenance?.cited_in_answer),
        dropped: Boolean(chunk.provenance?.was_dropped),
      },
      trace_flags: chunk.trace_flags || [],
      trace_provenance: chunk.provenance || {},
      ranks: chunk.ranks || {},
      scores: chunk.scores || {},
      reasons: chunk.reasons || {},
      explanation: chunk.explanation || '',
    })),
    {
      id: answerId,
      type: 'answer',
      label: 'Assistant answer',
      text_preview: answer.text_preview || '',
      model: answer.model || '',
      provider: answer.provider || '',
      summary: response.summary || {},
      trace_version: response.trace_version || 'v1-fallback',
      partial: Boolean(response.partial),
      partial_reason: response.partial_reason || '',
      assistant_message_id: response.assistant_message_id || response.message_id || null,
    },
  ];
}

// Edge type priority: higher = more significant stage in the pipeline
const EDGE_PRIORITY = {
  cited: 6,
  final_context: 5,
  context_selected: 4,
  reranked: 3,
  graph_added: 2,
  retrieved: 1,
  dropped: 0,
};

function buildEdgesFromV2Chunks(response) {
  const queryId = response.query?.id || `query:${response.user_message_id || 'latest'}`;
  const answerId = response.answer?.id || `answer:${response.assistant_message_id || response.message_id || 'latest'}`;

  // Collect ONE edge per direction per chunk, keeping the highest-priority type.
  // This prevents multiple overlapping arrows between the same node pair.
  const queryToChunk = new Map();  // chunkId → best edge type
  const chunkToAnswer = new Map(); // chunkId → best edge type

  for (const chunk of response.chunks || []) {
    const chunkId = `chunk:${chunk.id || chunk.chunk_id || chunk.relative_path}`;
    const provenance = chunk.provenance || {};
    const reachedAnswer = provenance.used_in_context || provenance.shown_as_final_source || provenance.cited_in_answer;

    // Determine the best query→chunk edge type
    let bestInbound = null;
    if (provenance.was_dropped && !reachedAnswer) {
      bestInbound = 'dropped';
    } else if (provenance.used_in_context) {
      bestInbound = 'context_selected';
    } else if (provenance.was_graph_added) {
      bestInbound = 'graph_added';
    } else if (provenance.was_retrieved) {
      bestInbound = 'retrieved';
    }
    if (bestInbound) {
      const existing = queryToChunk.get(chunkId);
      if (!existing || (EDGE_PRIORITY[bestInbound] || 0) > (EDGE_PRIORITY[existing] || 0)) {
        queryToChunk.set(chunkId, bestInbound);
      }
    }

    // Determine the best chunk→answer edge type (skip if dropped)
    if (provenance.was_dropped && !reachedAnswer) continue;

    let bestOutbound = null;
    if (provenance.cited_in_answer) {
      bestOutbound = 'cited';
    } else if (provenance.shown_as_final_source) {
      bestOutbound = 'final_context';
    } else if (provenance.used_in_context) {
      bestOutbound = 'context_selected';
    }
    if (bestOutbound) {
      const existing = chunkToAnswer.get(chunkId);
      if (!existing || (EDGE_PRIORITY[bestOutbound] || 0) > (EDGE_PRIORITY[existing] || 0)) {
        chunkToAnswer.set(chunkId, bestOutbound);
      }
    }
  }

  // Build final edge list — exactly one edge per direction per chunk
  const edges = [];
  for (const [chunkId, type] of queryToChunk) {
    edges.push({ source: queryId, target: chunkId, type });
  }
  for (const [chunkId, type] of chunkToAnswer) {
    edges.push({ source: chunkId, target: answerId, type });
  }
  return edges;
}
