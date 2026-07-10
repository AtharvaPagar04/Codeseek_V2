import test from 'node:test';
import assert from 'node:assert/strict';

import { classifySource, groupSources } from './sourceCards.js';

test('classifySource maps primary implementation code file', () => {
  const res = classifySource({
    relative_path: 'backend/retrieval/main.py',
    symbol_name: 'run_query',
    start_line: 12,
    end_line: 45,
    expansion_type: 'primary'
  });

  assert.equal(res.role, 'Primary implementation');
  assert.equal(res.badge, 'Code');
  assert.equal(res.label, 'backend/retrieval/main.py :: run_query');
  assert.equal(res.lines, '12-45');
});

test('classifySource maps supporting implementation code file', () => {
  const res = classifySource({
    relative_path: 'backend/retrieval/main.py',
    symbol_name: 'run_query',
    start_line: 12,
    end_line: 45,
    expansion_type: 'sibling'
  });

  assert.equal(res.role, 'Supporting implementation');
  assert.equal(res.badge, 'Code');
  assert.equal(res.lines, '12-45');
});

test('classifySource maps documentation file', () => {
  const res = classifySource({
    relative_path: 'docs/product/repo_freshness.md'
  });

  assert.equal(res.role, 'Documentation');
  assert.equal(res.badge, 'Docs');
});

test('classifySource maps test file', () => {
  const res = classifySource({
    relative_path: 'backend/tests/test_code_answers.py'
  });

  assert.equal(res.role, 'Tests');
  assert.equal(res.badge, 'Test');
});

test('classifySource maps config file', () => {
  const res = classifySource({
    relative_path: 'package.json'
  });

  assert.equal(res.role, 'Config');
  assert.equal(res.badge, 'Config');
});

test('groupSources groups and deduplicates sources correctly', () => {
  const sources = [
    {
      relative_path: 'backend/retrieval/main.py',
      symbol_name: 'run_query',
      start_line: 12,
      end_line: 45,
      expansion_type: 'primary'
    },
    // Duplicate of the above
    {
      relative_path: 'backend/retrieval/main.py',
      symbol_name: 'run_query',
      start_line: 12,
      end_line: 45,
      expansion_type: 'primary'
    },
    {
      relative_path: 'docs/product/repo_freshness.md'
    }
  ];

  const grouped = groupSources(sources);
  assert.equal(grouped['Primary implementation'].length, 1);
  assert.equal(grouped['Documentation'].length, 1);
  assert.equal(grouped['Tests'].length, 0);
});
