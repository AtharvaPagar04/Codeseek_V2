import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildNodeDetailsRenderModel,
  graphNodePath,
  graphNodeTitle,
  isFileNode,
  lineRangeLabel,
  normalizeNodeDetailsResponse,
  normalizeNodeDetailSymbols,
  symbolDescription,
  symbolKindLabel,
} from './nodeDetailsFormat.js';

test('isFileNode normalizes frontend and backend node type fields', () => {
  assert.equal(isFileNode({ type: 'file' }), true);
  assert.equal(isFileNode({ type: 'File' }), true);
  assert.equal(isFileNode({ node_type: 'file' }), true);
  assert.equal(isFileNode({ type: 'folder' }), false);
});

test('lineRangeLabel formats file symbol line ranges', () => {
  assert.equal(lineRangeLabel({ start_line: 4, end_line: 149 }), 'Lines 4-149');
  assert.equal(lineRangeLabel({ start_line: 12, end_line: 12 }), 'Line 12');
  assert.equal(lineRangeLabel({}), '');
});

test('symbolKindLabel formats chunk kind badges', () => {
  assert.equal(symbolKindLabel({ kind: 'ui_component' }), 'Ui Component');
  assert.equal(symbolKindLabel({ kind: 'function' }), 'Function');
});

test('symbolDescription uses safe fallback', () => {
  assert.equal(symbolDescription({ description: ' Renders project cards. ' }), 'Renders project cards.');
  assert.equal(symbolDescription({ description: '' }), 'No description available.');
});

test('normalizeNodeDetailSymbols prepares display fields', () => {
  const symbols = normalizeNodeDetailSymbols([
    {
      chunk_id: 'chunk-1',
      name: 'Projects',
      kind: 'component',
      description: 'Renders project cards.',
      start_line: 4,
      end_line: 149,
    },
    null,
    {},
  ]);

  assert.equal(symbols.length, 1);
  assert.equal(symbols[0].displayName, 'Projects');
  assert.equal(symbols[0].kindLabel, 'Component');
  assert.equal(symbols[0].lineLabel, 'Lines 4-149');
  assert.equal(symbols[0].displayDescription, 'Renders project cards.');
});

test('normalizeNodeDetailSymbols does not retain raw code fields', () => {
  const symbols = normalizeNodeDetailSymbols([
    {
      chunk_id: 'chunk-1',
      name: 'Projects',
      kind: 'component',
      description: 'Renders project cards.',
      content: 'function Projects() { return null }',
      content_excerpt: 'function Projects() { return null }',
    },
  ]);

  assert.equal(symbols.length, 1);
  assert.equal('content' in symbols[0], false);
  assert.equal('content_excerpt' in symbols[0], false);
});

test('normalizeNodeDetailsResponse maps backend symbols into major blocks', () => {
  const model = normalizeNodeDetailsResponse({
    status: 'ready',
    node: { id: 'n-file', type: 'file', path: 'src/components/Projects.tsx' },
    summary: { inbound_count: 2, outbound_count: 5, symbol_count: 1 },
    symbols: [
      {
        chunk_id: 'chunk-projects',
        name: 'Projects',
        kind: 'component',
        description: 'Renders project cards from project data.',
        start_line: 4,
        end_line: 149,
        label: 'ui_component',
      },
    ],
    connections: { imports: [], imported_by: [], defines: [] },
  });

  assert.equal(model.status, 'ready');
  assert.equal(model.majorBlocks.length, 1);
  assert.equal(model.majorBlocks[0].displayName, 'Projects');
  assert.equal(model.majorBlocks[0].displayDescription, 'Renders project cards from project data.');
});

test('normalizeNodeDetailsResponse handles empty symbols safely', () => {
  const model = normalizeNodeDetailsResponse({
    status: 'ready',
    node: { id: 'n-file', type: 'file' },
    symbols: [],
    message: 'No symbol descriptions found for this file.',
  });

  assert.deepEqual(model.majorBlocks, []);
  assert.equal(model.message, 'No symbol descriptions found for this file.');
});

test('buildNodeDetailsRenderModel keeps selected file visible during loading', () => {
  const model = buildNodeDetailsRenderModel(
    { id: 'n-file', type: 'file', label: 'Projects.tsx', path: 'src/components/Projects.tsx' },
    { status: 'loading', data: null }
  );

  assert.equal(model.hasSelection, true);
  assert.equal(model.isFile, true);
  assert.equal(model.title, 'Projects.tsx');
  assert.equal(model.path, 'src/components/Projects.tsx');
  assert.equal(model.status, 'loading');
});

test('graph node title and path prefer display-safe fields', () => {
  assert.equal(graphNodeTitle({ label: 'Projects.tsx', path: 'src/components/Projects.tsx' }), 'Projects.tsx');
  assert.equal(graphNodeTitle({ name: 'Projects' }), 'Projects');
  assert.equal(graphNodePath({ relative_path: 'src/app/page.tsx' }), 'src/app/page.tsx');
});
