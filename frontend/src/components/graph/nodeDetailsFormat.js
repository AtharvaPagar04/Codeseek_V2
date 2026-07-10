export function isFileNode(node) {
  return String(node?.type || node?.node_type || '').toLowerCase() === 'file';
}

export function lineRangeLabel(symbol) {
  const start = symbol?.start_line;
  const end = symbol?.end_line;
  if (start && end && start !== end) return `Lines ${start}-${end}`;
  if (start) return `Line ${start}`;
  return '';
}

export function symbolKindLabel(symbol) {
  const raw = symbol?.kind || symbol?.label || 'symbol';
  return String(raw).replace(/[_-]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

export function symbolDescription(symbol) {
  const text = String(symbol?.description || '').trim();
  return text || 'No description available.';
}

export function normalizeNodeDetailSymbols(symbols = []) {
  if (!Array.isArray(symbols)) return [];
  return symbols
    .filter((symbol) => symbol && (symbol.name || symbol.qualified_name || symbol.chunk_id))
    .map((symbol) => ({
      displayName: symbol.name || symbol.qualified_name || symbol.chunk_id,
      kindLabel: symbolKindLabel(symbol),
      lineLabel: lineRangeLabel(symbol),
      displayDescription: symbolDescription(symbol),
      chunk_id: symbol.chunk_id || null,
      name: symbol.name || '',
      qualified_name: symbol.qualified_name || '',
      kind: symbol.kind || symbol.label || 'symbol',
      start_line: symbol.start_line ?? null,
      end_line: symbol.end_line ?? null,
      label: symbol.label || '',
      labels: Array.isArray(symbol.labels) ? symbol.labels : [],
    }));
}

export function graphNodeTitle(node) {
  return node?.label || node?.name || node?.path || node?.relative_path || node?.id || 'Selected node';
}

export function graphNodePath(node) {
  return node?.path || node?.relative_path || '';
}

export function normalizeNodeDetailsResponse(details) {
  const safe = details && typeof details === 'object' ? details : {};
  const symbols = normalizeNodeDetailSymbols(safe.symbols || []);
  return {
    status: safe.status || '',
    node: safe.node || null,
    summary: safe.summary || {},
    majorBlocks: symbols,
    connections: safe.connections || {},
    message: safe.message || '',
  };
}

export function buildNodeDetailsRenderModel(selectedNode, detailsState = {}) {
  const details = normalizeNodeDetailsResponse(detailsState.data);
  const node = details.node || selectedNode || null;
  const file = isFileNode(node) || isFileNode(selectedNode);
  return {
    hasSelection: Boolean(selectedNode || details.node),
    isFile: file,
    title: graphNodeTitle(node || selectedNode),
    path: graphNodePath(node || selectedNode),
    type: String(node?.type || node?.node_type || selectedNode?.type || selectedNode?.node_type || ''),
    symbolName: node?.symbol_name || selectedNode?.symbol_name || '',
    importance: selectedNode?.importance,
    status: detailsState.status || 'idle',
    detailsStatus: details.status,
    summary: details.summary,
    majorBlocks: details.majorBlocks,
    message: details.message,
  };
}
