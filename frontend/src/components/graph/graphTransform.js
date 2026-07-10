/**
 * graphTransform.js — transforms backend graph API response into
 * react-force-graph-compatible data, applies filters, caps sizes,
 * and marks visual properties on nodes/edges.
 */

const DEFAULT_MAX_NODES = 250;
const DEFAULT_MAX_EDGES = 500;

/**
 * Infer a short display label from a node.
 */
export function inferLabel(node) {
  if (node.label) return node.label;
  if (node.symbol_name) return node.symbol_name;
  if (node.path) {
    const parts = node.path.split('/');
    return parts[parts.length - 1] || node.path;
  }
  return node.id || '?';
}

/**
 * Normalize a single node from backend format.
 */
export function normalizeNode(raw) {
  const type = (raw.type || 'unknown').toLowerCase();
  return {
    id: raw.id,
    label: inferLabel(raw),
    type,
    path: raw.path || null,
    symbol_name: raw.symbol_name || null,
    size: raw.size || sizeForType(type),
    importance: raw.importance ?? 0.5,
    is_retrieved: Boolean(raw.is_retrieved),
    is_graph_active: Boolean(raw.is_graph_active),
    metadata: raw.metadata || {},
  };
}

function sizeForType(type) {
  switch (type) {
    case 'folder':   return 7;
    case 'file':     return 5;
    case 'symbol':   return 4;
    case 'external': return 4;
    default:         return 3;
  }
}

/**
 * Normalize a single edge from backend format.
 */
export function normalizeEdge(raw) {
  const type = (raw.type || 'unknown').toLowerCase();
  const isImport = type === 'imports';
  const source = isImport ? raw.target : raw.source;
  const target = isImport ? raw.source : raw.target;
  return {
    id: raw.id || `${source}→${target}`,
    source,
    target,
    type,
    weight: raw.weight ?? 1,
    metadata: raw.metadata || {},
  };
}

/**
 * Transform full backend response into renderer-ready graph data.
 *
 * @param {Object} response - backend graph API response
 * @param {Object} options
 * @param {string[]} options.nodeTypeFilter - show only these node types (empty = all)
 * @param {string[]} options.edgeTypeFilter - show only these edge types (empty = all)
 * @param {string} options.searchQuery - filter nodes whose label/path contains this
 * @param {number} options.maxNodes
 * @param {number} options.maxEdges
 * @returns {{ nodes: Object[], links: Object[], summary: Object }}
 */
export function transformGraphData(response, options = {}) {
  if (!response || !response.nodes) {
    return { nodes: [], links: [], summary: emptySummary() };
  }

  const {
    nodeTypeFilter = [],
    edgeTypeFilter = [],
    searchQuery = '',
    maxNodes = DEFAULT_MAX_NODES,
    maxEdges = DEFAULT_MAX_EDGES,
  } = options;

  // 1. Normalize and deduplicate nodes by ID
  const normalizedNodes = response.nodes.map(normalizeNode);
  const nodeById = new Map();
  for (const node of normalizedNodes) {
    if (node) nodeById.set(node.id, node);
  }
  let nodes = Array.from(nodeById.values());

  // 2. Apply node type filter
  if (nodeTypeFilter.length > 0) {
    const allowed = new Set(nodeTypeFilter.map((t) => t.toLowerCase()));
    nodes = nodes.filter((n) => allowed.has(n.type));
  }

  // 3. Apply search filter
  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase();
    nodes = nodes.filter(
      (n) =>
        n.label.toLowerCase().includes(q) ||
        (n.path && n.path.toLowerCase().includes(q)) ||
        (n.symbol_name && n.symbol_name.toLowerCase().includes(q))
    );
  }

  // 4. Cap nodes
  if (nodes.length > maxNodes) {
    nodes.sort((a, b) => b.importance - a.importance);
    nodes = nodes.slice(0, maxNodes);
  }

  // 5. Build node id set for edge filtering
  const nodeIds = new Set(nodes.map((n) => n.id));

  // 6. Normalize edges
  let links = (response.edges || []).map(normalizeEdge);

  // 7. Remove dangling edges
  links = links.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

  // 8. Apply edge type filter
  if (edgeTypeFilter.length > 0) {
    const allowed = new Set(edgeTypeFilter.map((t) => t.toLowerCase()));
    links = links.filter((e) => allowed.has(e.type));
  }

  // 9. Cap edges
  if (links.length > maxEdges) {
    links.sort((a, b) => b.weight - a.weight);
    links = links.slice(0, maxEdges);
  }

  // 9.5 Compute dynamic node sizes based on active layout degree
  const degrees = {};
  for (const link of links) {
    const src = typeof link.source === 'object' ? link.source.id : link.source;
    const tgt = typeof link.target === 'object' ? link.target.id : link.target;
    if (src) degrees[src] = (degrees[src] || 0) + 1;
    if (tgt) degrees[tgt] = (degrees[tgt] || 0) + 1;
  }

  nodes = nodes.map((node) => {
    const deg = degrees[node.id] || 0;
    const base = node.size || sizeForType(node.type);
    const scale = 1.0 + Math.min(Math.log1p(deg) * 0.35, 1.2);
    return {
      ...node,
      size: base * scale,
      degree: deg,
    };
  });

  // 10. Build summary
  const summary = buildSummary(nodes, links);

  return { nodes, links, summary };
}

function emptySummary() {
  return {
    node_count: 0,
    edge_count: 0,
    node_types: {},
    edge_types: {},
  };
}

function buildSummary(nodes, links) {
  const nodeTypes = {};
  for (const n of nodes) {
    nodeTypes[n.type] = (nodeTypes[n.type] || 0) + 1;
  }
  const edgeTypes = {};
  for (const e of links) {
    edgeTypes[e.type] = (edgeTypes[e.type] || 0) + 1;
  }
  return {
    node_count: nodes.length,
    edge_count: links.length,
    node_types: nodeTypes,
    edge_types: edgeTypes,
  };
}

/**
 * Get neighbor node IDs for a given node (1-hop).
 */
export function getNeighborIds(nodeId, links) {
  const ids = new Set();
  for (const link of links) {
    const src = typeof link.source === 'object' ? link.source.id : link.source;
    const tgt = typeof link.target === 'object' ? link.target.id : link.target;
    if (src === nodeId) ids.add(tgt);
    if (tgt === nodeId) ids.add(src);
  }
  return ids;
}

/**
 * Count inbound and outbound edges for a node.
 */
export function getNodeDegree(nodeId, links) {
  let inbound = 0;
  let outbound = 0;
  for (const link of links) {
    const src = typeof link.source === 'object' ? link.source.id : link.source;
    const tgt = typeof link.target === 'object' ? link.target.id : link.target;
    if (src === nodeId) outbound++;
    if (tgt === nodeId) inbound++;
  }
  return { inbound, outbound };
}
