import test from 'node:test';
import assert from 'node:assert/strict';

import {
  normalizeTraceNode,
  transformRetrievalTraceData,
} from './retrievalTraceTransform.js';

test('normalizeTraceNode maps stage flags to visual flags', () => {
  const node = normalizeTraceNode({
    id: 'chunk:1',
    type: 'chunk',
    label: 'Projects',
    chunk_id: '1',
    stage_flags: { retrieved: true, graph_added: true, final: true, cited: true },
  });

  assert.equal(node.is_retrieved, true);
  assert.equal(node.is_graph_active, true);
  assert.equal(node.is_trace_final, true);
  assert.equal(node.is_trace_cited, true);
  assert.equal(node.type, 'chunk');
});

test('transformRetrievalTraceData maps query chunk answer nodes and edges', () => {
  const graph = transformRetrievalTraceData({
    status: 'ready',
    nodes: [
      { id: 'query:q1', type: 'query', label: 'Question' },
      {
        id: 'chunk:c1',
        type: 'chunk',
        label: 'Projects',
        chunk_id: 'c1',
        path: 'src/components/Projects.tsx',
        stage_flags: { retrieved: true, final: true, cited: true },
      },
      { id: 'answer:a1', type: 'answer', label: 'Assistant answer' },
    ],
    edges: [
      { id: 'e1', source: 'query:q1', target: 'chunk:c1', type: 'retrieved' },
      { id: 'e2', source: 'chunk:c1', target: 'answer:a1', type: 'cited' },
    ],
  });

  assert.equal(graph.nodes.length, 3);
  assert.equal(graph.links.length, 2);
  assert.equal(graph.summary.cited_count, 1);
  assert.equal(graph.summary.edge_types.cited, 1);
});

test('transformRetrievalTraceData filters chunks by stage edge filters', () => {
  const graph = transformRetrievalTraceData(
    {
      status: 'ready',
      nodes: [
        { id: 'query:q1', type: 'query', label: 'Question' },
        { id: 'chunk:retrieved', type: 'chunk', label: 'Retrieved', stage_flags: { retrieved: true } },
        { id: 'chunk:graph', type: 'chunk', label: 'Graph', stage_flags: { graph_added: true } },
        { id: 'answer:a1', type: 'answer', label: 'Assistant answer' },
      ],
      edges: [
        { source: 'query:q1', target: 'chunk:retrieved', type: 'retrieved' },
        { source: 'query:q1', target: 'chunk:graph', type: 'graph_added' },
        { source: 'chunk:graph', target: 'answer:a1', type: 'final_context' },
      ],
    },
    { edgeTypeFilter: ['graph_added'] }
  );

  const ids = graph.nodes.map((node) => node.id);
  assert.ok(ids.includes('chunk:graph'));
  assert.equal(ids.includes('chunk:retrieved'), false);
  assert.equal(graph.links.length, 1);
  assert.equal(graph.links[0].type, 'graph_added');
});

test('transformRetrievalTraceData handles empty payload safely', () => {
  const graph = transformRetrievalTraceData({
    status: 'empty',
    message: 'Ask a question first.',
    nodes: [],
    edges: [],
  });

  assert.deepEqual(graph.nodes, []);
  assert.deepEqual(graph.links, []);
  assert.equal(graph.summary.node_count, 0);
  assert.equal(graph.summary.message, 'Ask a question first.');
});

test('transformRetrievalTraceData handles missing optional metadata safely', () => {
  const graph = transformRetrievalTraceData({
    nodes: [
      { id: 'query:q1', type: 'query' },
      { id: 'chunk:c1', type: 'chunk' },
      { id: 'answer:a1', type: 'answer' },
    ],
    edges: [{ source: 'query:q1', target: 'chunk:c1', type: 'retrieved' }],
  });

  const chunk = graph.nodes.find((node) => node.id === 'chunk:c1');
  assert.equal(chunk.label, 'chunk:c1');
  assert.equal(chunk.path, null);
  assert.equal(graph.summary.retrieved_count, 0);
});

test('transformRetrievalTraceData normalizes V2 persisted trace chunks', () => {
  const graph = transformRetrievalTraceData({
    status: 'ready',
    trace_version: 'v2',
    partial: false,
    query: { id: 'query:u1', text: 'How do links render?' },
    answer: { id: 'answer:a1', text_preview: 'Projects renders links.' },
    summary: {
      retrieved_count: 3,
      graph_added_count: 1,
      reranked_count: 4,
      context_selected_count: 2,
      final_source_count: 2,
      cited_count: 1,
      dropped_count: 2,
    },
    chunks: [
      {
        id: 'chunk-b',
        chunk_id: 'chunk-b',
        label: 'Projects',
        relative_path: 'src/components/Projects.tsx',
        symbol_name: 'Projects',
        provenance: {
          was_retrieved: true,
          was_graph_added: false,
          survived_rerank: true,
          used_in_context: true,
          shown_as_final_source: true,
          cited_in_answer: false,
          was_dropped: false,
        },
        ranks: { retrieved: 2, reranked: 1, context_order: 1, display: 1 },
        scores: { retrieved: 0.8, reranked: 0.95 },
        trace_flags: ['retrieved', 'reranked_in', 'context_selected', 'final_source'],
        explanation: 'Base retrieval rank #2 -> selected into context',
      },
      {
        id: 'chunk-d',
        chunk_id: 'chunk-d',
        label: 'project data',
        relative_path: 'src/lib/data.ts',
        symbol_name: 'projects',
        provenance: {
          was_retrieved: false,
          was_graph_added: true,
          survived_rerank: true,
          used_in_context: true,
          shown_as_final_source: true,
          cited_in_answer: true,
          was_dropped: false,
        },
        ranks: { graph: 1, reranked: 2, context_order: 2, display: 2, citation: 1 },
        scores: { graph: 98, reranked: 0.93 },
        reasons: {
          graph_anchor_path: 'src/components/Projects.tsx',
          graph_score_reasons: ['query_match:projects'],
        },
        trace_flags: ['graph_added', 'reranked_in', 'context_selected', 'final_source', 'cited'],
      },
      {
        id: 'chunk-c',
        chunk_id: 'chunk-c',
        label: 'About',
        relative_path: 'src/components/About.tsx',
        symbol_name: 'About',
        provenance: {
          was_retrieved: true,
          was_graph_added: false,
          survived_rerank: false,
          used_in_context: false,
          shown_as_final_source: false,
          cited_in_answer: false,
          was_dropped: true,
        },
        reasons: { drop_reason: 'low_query_match' },
        trace_flags: ['retrieved', 'dropped'],
      },
    ],
  });

  assert.equal(graph.meta.trace_version, 'v2');
  assert.equal(graph.summary.context_selected_count, 2);
  assert.equal(graph.summary.dropped_count, 2);
  const dataNode = graph.nodes.find((node) => node.chunk_id === 'chunk-d');
  assert.equal(dataNode.is_graph_active, true);
  assert.equal(dataNode.is_trace_cited, true);
  assert.equal(dataNode.flags.contextSelected, true);
  assert.equal(dataNode.reasons.graph_anchor_path, 'src/components/Projects.tsx');
  const droppedNode = graph.nodes.find((node) => node.chunk_id === 'chunk-c');
  assert.equal(droppedNode.is_trace_dropped, true);
  assert.equal(droppedNode.reasons.drop_reason, 'low_query_match');
  assert.ok(graph.links.some((edge) => edge.type === 'dropped'));
  assert.ok(graph.chunksById['chunk-d']);
});

test('transformRetrievalTraceData filters V2 dropped candidates', () => {
  const graph = transformRetrievalTraceData(
    {
      status: 'ready',
      trace_version: 'v2',
      chunks: [
        {
          id: 'kept',
          chunk_id: 'kept',
          label: 'Kept',
          provenance: { used_in_context: true, was_dropped: false },
        },
        {
          id: 'dropped',
          chunk_id: 'dropped',
          label: 'Dropped',
          provenance: { was_retrieved: true, was_dropped: true },
        },
      ],
    },
    { edgeTypeFilter: ['dropped'] }
  );

  assert.ok(graph.nodes.some((node) => node.chunk_id === 'dropped'));
  assert.equal(graph.nodes.some((node) => node.chunk_id === 'kept'), false);
});

test('transformRetrievalTraceData exposes partial fallback metadata', () => {
  const graph = transformRetrievalTraceData({
    status: 'partial',
    trace_version: 'v1-fallback',
    partial: true,
    partial_reason: 'persisted_trace_unavailable',
    assistant_message_id: 'assistant-1',
    user_message_id: 'user-1',
    nodes: [],
    edges: [],
  });

  assert.equal(graph.summary.partial, true);
  assert.equal(graph.summary.trace_version, 'v1-fallback');
  assert.equal(graph.meta.partial_reason, 'persisted_trace_unavailable');
  assert.equal(graph.traceVersion, 'v1-fallback');
  assert.equal(graph.partial, true);
  assert.equal(graph.assistantMessageId, 'assistant-1');
  assert.equal(graph.userMessageId, 'user-1');
});

test('transformRetrievalTraceData enriches query and answer nodes with trace metadata', () => {
  const graph = transformRetrievalTraceData({
    status: 'ready',
    trace_version: 'v2',
    partial: false,
    assistant_message_id: 'assistant-2',
    user_message_id: 'user-2',
    request: { graph_retrieval_mode: 'graph_assist', intent: 'EXPLANATION' },
    summary: { retrieved_count: 1, cited_count: 1 },
    nodes: [
      { id: 'query:user-2', type: 'query', label: 'Question', text: 'Question' },
      { id: 'answer:assistant-2', type: 'answer', label: 'Assistant answer', text_preview: 'Answer' },
      { id: 'chunk:c1', type: 'chunk', label: 'Chunk', stage_flags: { retrieved: true, cited: true } },
    ],
    edges: [
      { source: 'query:user-2', target: 'chunk:c1', type: 'retrieved' },
      { source: 'chunk:c1', target: 'answer:assistant-2', type: 'cited' },
    ],
  });

  const query = graph.nodes.find((node) => node.type === 'query');
  const answer = graph.nodes.find((node) => node.type === 'answer');

  assert.equal(query.request.graph_retrieval_mode, 'graph_assist');
  assert.equal(query.user_message_id, 'user-2');
  assert.equal(answer.assistant_message_id, 'assistant-2');
  assert.equal(answer.trace_version, 'v2');
  assert.equal(answer.summary.cited_count, 1);
});

test('transformRetrievalTraceData unifies duplicate chunk nodes with different ID representations', () => {
  const graph = transformRetrievalTraceData({
    status: 'ready',
    trace_version: 'v2',
    nodes: [
      { id: 'query:q1', type: 'query', label: 'Question' },
      {
        id: 'chunk:c1',
        type: 'chunk',
        label: 'Projects',
        chunk_id: 'c1',
        path: 'src/components/Projects.tsx',
        symbol_name: 'Projects',
        start_line: 10,
        stage_flags: { retrieved: true },
      },
      {
        id: 'chunk:path:src/components/Projects.tsx:Projects:10',
        type: 'chunk',
        label: 'Projects',
        chunk_id: '',
        path: 'src/components/Projects.tsx',
        symbol_name: 'Projects',
        start_line: 10,
        stage_flags: { cited: true },
      },
      { id: 'answer:a1', type: 'answer', label: 'Assistant answer' },
    ],
    edges: [
      { id: 'e1', source: 'query:q1', target: 'chunk:path:src/components/Projects.tsx:Projects:10', type: 'retrieved' },
      { id: 'e2', source: 'chunk:c1', target: 'answer:a1', type: 'cited' },
    ],
  });

  // Both should be unified into the preferred 'chunk:c1' ID
  assert.equal(graph.nodes.length, 3);
  const chunkNode = graph.nodes.find((node) => node.type === 'chunk');
  assert.equal(chunkNode.id, 'chunk:c1');
  assert.equal(chunkNode.is_retrieved, true);
  assert.equal(chunkNode.is_trace_cited, true);

  // Both edges should point to the unified node 'chunk:c1'
  assert.equal(graph.links.length, 2);
  assert.ok(graph.links.every((link) => link.source === 'query:q1' || link.target === 'answer:a1'));
  assert.ok(graph.links.every((link) => link.source === 'chunk:c1' || link.target === 'chunk:c1'));
});
