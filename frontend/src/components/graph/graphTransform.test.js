import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import {
  inferLabel,
  normalizeNode,
  normalizeEdge,
  transformGraphData,
  getNeighborIds,
  getNodeDegree,
} from './graphTransform.js';

test('inferLabel uses label, then symbol_name, then filename from path', () => {
  assert.equal(inferLabel({ label: 'Foo' }), 'Foo');
  assert.equal(inferLabel({ symbol_name: 'bar' }), 'bar');
  assert.equal(inferLabel({ path: 'src/components/Projects.tsx' }), 'Projects.tsx');
  assert.equal(inferLabel({ id: 'node-1' }), 'node-1');
  assert.equal(inferLabel({}), '?');
});

test('normalizeNode maps backend fields correctly', () => {
  const node = normalizeNode({
    id: 'n1',
    type: 'FILE',
    path: 'src/App.tsx',
    is_retrieved: true,
    importance: 0.9,
  });
  assert.equal(node.id, 'n1');
  assert.equal(node.type, 'file');
  assert.equal(node.label, 'App.tsx');
  assert.equal(node.is_retrieved, true);
  assert.equal(node.is_graph_active, false);
  assert.equal(node.importance, 0.9);
});

test('normalizeNode uses sensible defaults for missing fields', () => {
  const node = normalizeNode({ id: 'x' });
  assert.equal(node.type, 'unknown');
  assert.equal(node.size, 3);
  assert.equal(node.importance, 0.5);
  assert.equal(node.is_retrieved, false);
  assert.deepEqual(node.metadata, {});
});

test('normalizeEdge maps backend fields and generates fallback id', () => {
  // imports edges swap source and target for arrow direction
  const edge = normalizeEdge({ source: 'a', target: 'b', type: 'Imports' });
  assert.equal(edge.id, 'b→a');
  assert.equal(edge.source, 'b');
  assert.equal(edge.target, 'a');
  assert.equal(edge.type, 'imports');
  assert.equal(edge.weight, 1);
});

test('transformGraphData returns empty result for null input', () => {
  const result = transformGraphData(null);
  assert.deepEqual(result.nodes, []);
  assert.deepEqual(result.links, []);
  assert.equal(result.summary.node_count, 0);
});

test('transformGraphData normalizes and filters nodes by type', () => {
  const response = {
    nodes: [
      { id: 'f1', type: 'file', path: 'src/a.ts' },
      { id: 's1', type: 'symbol', symbol_name: 'foo' },
      { id: 'd1', type: 'folder', path: 'src' },
    ],
    edges: [
      { source: 'f1', target: 's1', type: 'defines' },
      { source: 'd1', target: 'f1', type: 'contains' },
    ],
  };

  const result = transformGraphData(response, { nodeTypeFilter: ['file', 'symbol'] });
  assert.equal(result.nodes.length, 2);
  assert.equal(result.summary.node_types.folder, undefined);
  // folder removed → contains edge should be dangling and removed
  assert.equal(result.links.filter((l) => l.type === 'contains').length, 0);
});

test('transformGraphData removes dangling edges', () => {
  const response = {
    nodes: [{ id: 'a', type: 'file', path: 'a.ts' }],
    edges: [
      { source: 'a', target: 'b', type: 'imports' }, // b doesn't exist
    ],
  };
  const result = transformGraphData(response);
  assert.equal(result.links.length, 0);
});

test('transformGraphData filters by search query', () => {
  const response = {
    nodes: [
      { id: 'f1', type: 'file', path: 'src/Projects.tsx' },
      { id: 'f2', type: 'file', path: 'src/App.tsx' },
    ],
    edges: [],
  };
  const result = transformGraphData(response, { searchQuery: 'project' });
  assert.equal(result.nodes.length, 1);
  assert.equal(result.nodes[0].label, 'Projects.tsx');
});

test('transformGraphData caps nodes and edges', () => {
  const nodes = Array.from({ length: 10 }, (_, i) => ({
    id: `n${i}`,
    type: 'file',
    path: `f${i}.ts`,
    importance: i / 10,
  }));
  const edges = [];
  // create edges between consecutive nodes
  for (let i = 0; i < 9; i++) {
    edges.push({ source: `n${i}`, target: `n${i + 1}`, type: 'imports' });
  }
  const result = transformGraphData({ nodes, edges }, { maxNodes: 5, maxEdges: 3 });
  assert.ok(result.nodes.length <= 5);
  assert.ok(result.links.length <= 3);
});

test('transformGraphData filters edges by type', () => {
  const response = {
    nodes: [
      { id: 'a', type: 'file', path: 'a.ts' },
      { id: 'b', type: 'file', path: 'b.ts' },
    ],
    edges: [
      { source: 'a', target: 'b', type: 'imports' },
      { source: 'a', target: 'b', type: 'defines' },
    ],
  };
  const result = transformGraphData(response, { edgeTypeFilter: ['imports'] });
  assert.equal(result.links.length, 1);
  assert.equal(result.links[0].type, 'imports');
});

test('transformGraphData builds correct summary', () => {
  const response = {
    nodes: [
      { id: 'a', type: 'file', path: 'a.ts' },
      { id: 'b', type: 'symbol', symbol_name: 'foo' },
    ],
    edges: [{ source: 'a', target: 'b', type: 'defines' }],
  };
  const { summary } = transformGraphData(response);
  assert.equal(summary.node_count, 2);
  assert.equal(summary.edge_count, 1);
  assert.equal(summary.node_types.file, 1);
  assert.equal(summary.node_types.symbol, 1);
  assert.equal(summary.edge_types.defines, 1);
});

test('getNeighborIds returns connected node IDs', () => {
  const links = [
    { source: 'a', target: 'b', type: 'imports' },
    { source: 'c', target: 'a', type: 'defines' },
    { source: 'x', target: 'y', type: 'imports' },
  ];
  const neighbors = getNeighborIds('a', links);
  assert.ok(neighbors.has('b'));
  assert.ok(neighbors.has('c'));
  assert.ok(!neighbors.has('x'));
});

test('getNodeDegree counts inbound and outbound', () => {
  const links = [
    { source: 'a', target: 'b', type: 'imports' },
    { source: 'c', target: 'a', type: 'defines' },
    { source: 'a', target: 'd', type: 'imports' },
  ];
  const { inbound, outbound } = getNodeDegree('a', links);
  assert.equal(inbound, 1);
  assert.equal(outbound, 2);
});
