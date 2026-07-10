import test from 'node:test';
import assert from 'node:assert/strict';

import { buildAnswerDiagnosticsRows, summarizeDiagnosticSource, sanitizeCredentialsInString } from './answerDiagnostics.js';

test('buildAnswerDiagnosticsRows keeps only safe display fields', () => {
  const rows = buildAnswerDiagnosticsRows({
    intent: 'CODE_REQUEST',
    primary_intent: 'CODE_REQUEST',
    response_mode: 'code_snippet',
    provider: 'local',
    model: 'qwen2.5-coder:3b-8k',
    routing_mode: 'local',
    context_tokens: 512,
    evidence_confidence: { level: 'strong', reason: 'matched route', count: 2 },
    source_filter: { selected_primary: 1, selected_expanded: 0, display_count: 1, reasoning_count: 2 },
    session_status: 'ready',
    session_error: '',
    validation: { valid: false, reasons: ['rebuilt_code_snippet'] },
    memory: {
      is_followup: false,
      topic_shift_detected: true,
      followup_confidence: 0.125,
      query_similarity: 0.0,
      keyword_overlap: 0.0,
      similarity_method: 'keyword_overlap',
      has_valid_referent: false,
      history_injected: false,
      history_turns_used: 0,
    },
    rewrite: {
      query_rewritten: false,
      rewrite_mode: 'none',
      rewrite_anchor: null,
    },
    retrieval: {
      previous_candidates_injected: 0,
      strong_new_entities: ['backend/retrieval/api_service.py', '_require_auth'],
      exact_hit: true,
      multi_layer_hit: true,
      top_score: 0.97,
      candidate_count: 4,
      retrieval_confidence: 'strong',
      low_confidence_gate: false,
    },
    graph_active: {
      enabled: true,
      requested_mode: 'graph_assist',
      effective_enabled: true,
      reason: 'added',
      added_count: 1,
      added_chunks: [
        {
          relative_path: 'src/components/Projects.tsx',
          chunk_id: 'projects',
        },
      ],
    },
    selected_sources: [
      {
        relative_path: 'backend/evals/run_safe_evals.py',
        symbol_name: 'main',
        start_line: 10,
        end_line: 48,
        api_key: 'secret',
        raw_prompt: 'hidden',
      },
    ],
    reasoning_sources: [
      {
        relative_path: 'backend/evals/run_safe_evals.py',
        symbol_name: 'get_tail',
        start_line: 50,
        end_line: 66,
      },
    ],
    rendered_sources: [
      {
        relative_path: 'backend/evals/run_safe_evals.py',
        symbol_name: 'main',
        start_line: 10,
        end_line: 48,
      },
    ],
  });

  assert.ok(rows.length > 0);
  assert.equal(rows[0].label, 'Intent');
  assert.ok(rows.some((row) => row.label === 'Validation'));
  assert.ok(rows.some((row) => row.label === 'Rendered sources'));
  assert.ok(rows.some((row) => row.label === 'Selected sources'));
  assert.ok(rows.some((row) => row.label === 'Reasoning sources'));
  assert.ok(rows.some((row) => row.label === 'Follow-up'));
  assert.ok(rows.some((row) => row.label === 'History injected'));
  assert.ok(rows.some((row) => row.label === 'Previous candidates injected'));
  assert.ok(rows.some((row) => row.label === 'Strong new entities'));
  assert.ok(rows.some((row) => row.label === 'Graph Assist'));
  assert.ok(rows.some((row) => row.label === 'Graph Assist added paths'));
  const renderedRow = rows.find((row) => row.label === 'Rendered sources');
  assert.equal(renderedRow.value[0], 'backend/evals/run_safe_evals.py :: main (L10–48)');
  const selectedRow = rows.find((row) => row.label === 'Selected sources');
  assert.equal(selectedRow.value[0], 'backend/evals/run_safe_evals.py :: main (L10–48)');
  const strongEntitiesRow = rows.find((row) => row.label === 'Strong new entities');
  assert.deepEqual(strongEntitiesRow.value, ['backend/retrieval/api_service.py', '_require_auth']);
  const graphAssistRow = rows.find((row) => row.label === 'Graph Assist');
  assert.equal(graphAssistRow.value, 'graph_assist · effective · 1 added · added');
  const graphAssistAddedRow = rows.find((row) => row.label === 'Graph Assist added paths');
  assert.deepEqual(graphAssistAddedRow.value, ['src/components/Projects.tsx']);
  assert.ok(rows.every((row) => JSON.stringify(row).indexOf('secret') === -1));
  assert.ok(rows.every((row) => JSON.stringify(row).indexOf('hidden') === -1));
});

test('buildAnswerDiagnosticsRows shows unavailable Graph Assist without crashing', () => {
  const rows = buildAnswerDiagnosticsRows({
    graph_active: {
      enabled: false,
      requested_mode: 'graph_assist',
      effective_enabled: false,
      reason: 'disabled',
      disabled_reason: 'disabled_by_server_config',
      added_count: 0,
      added_chunks: [],
    },
  });

  const summary = rows.find((row) => row.label === 'Graph Assist');
  const disabledReason = rows.find((row) => row.label === 'Graph Assist disabled reason');
  assert.equal(
    summary.value,
    'graph_assist · not effective · 0 added · disabled_by_server_config'
  );
  assert.equal(disabledReason.value, 'disabled_by_server_config');
});

test('summarizeDiagnosticSource handles missing fields safely', () => {
  assert.equal(summarizeDiagnosticSource(null), '');
  assert.equal(
    summarizeDiagnosticSource({
      relative_path: 'backend/evals/run_safe_evals.py',
      symbol_name: 'main',
      start_line: 10,
      end_line: 48,
    }),
    'backend/evals/run_safe_evals.py :: main (L10–48)'
  );
});

test('buildAnswerDiagnosticsRows includes sections, isAdvanced, and freshness metadata', () => {
  const rows = buildAnswerDiagnosticsRows({
    intent: 'CODE_REQUEST',
    provider: 'local',
    model: 'qwen2.5-coder:3b-8k',
    freshness: {
      status: 'branch_changed',
      indexed_branch: 'main',
      current_branch: 'feature-branch',
      branch_changed: true,
      indexed_commit_sha: 'commit123',
      current_commit_sha: 'commit456',
      dirty_worktree: false,
      checked_at: '2026-06-12T16:25:15+05:30'
    }
  });

  const intentRow = rows.find(r => r.label === 'Intent');
  assert.ok(intentRow);
  assert.equal(intentRow.section, 'Intent');
  assert.equal(intentRow.isAdvanced, false);

  const modelRow = rows.find(r => r.label === 'Model');
  assert.ok(modelRow);
  assert.equal(modelRow.section, 'Model');
  assert.equal(modelRow.isAdvanced, false);

  const freshnessStatusRow = rows.find(r => r.label === 'Freshness status');
  assert.ok(freshnessStatusRow);
  assert.equal(freshnessStatusRow.value, 'branch_changed');
  assert.equal(freshnessStatusRow.section, 'Freshness');
  assert.equal(freshnessStatusRow.isAdvanced, false);

  const indexedBranchRow = rows.find(r => r.label === 'Indexed branch');
  assert.ok(indexedBranchRow);
  assert.equal(indexedBranchRow.value, 'main');
  assert.equal(indexedBranchRow.isAdvanced, true);

  const currentBranchRow = rows.find(r => r.label === 'Current branch');
  assert.ok(currentBranchRow);
  assert.equal(currentBranchRow.value, 'feature-branch');
  assert.equal(currentBranchRow.isAdvanced, true);

  const branchChangedRow = rows.find(r => r.label === 'Branch changed');
  assert.ok(branchChangedRow);
  assert.equal(branchChangedRow.value, 'Yes');
  assert.equal(branchChangedRow.isAdvanced, true);

  const indexedCommitRow = rows.find(r => r.label === 'Indexed commit');
  assert.ok(indexedCommitRow);
  assert.equal(indexedCommitRow.value, 'commit123');
  assert.equal(indexedCommitRow.section, 'Freshness');
  assert.equal(indexedCommitRow.isAdvanced, true);
});

test('sanitizeCredentialsInString redacts bearer tokens and URL credentials', () => {
  assert.equal(sanitizeCredentialsInString('bearer ghp_123xyz'), 'Bearer [redacted]');
  assert.equal(sanitizeCredentialsInString('Bearer ghp_123xyz'), 'Bearer [redacted]');
  assert.equal(
    sanitizeCredentialsInString('https://ghp_abc123xyz@github.com/org/repo.git'),
    'https://[redacted]@github.com/org/repo.git'
  );
  assert.equal(
    sanitizeCredentialsInString('postgresql://postgres:mysecretpassword@localhost:5432/codeseek'),
    'postgresql://[redacted]:[redacted]@localhost:5432/codeseek'
  );
  
  // Test buildAnswerDiagnosticsRows with sensitive error message
  const rows = buildAnswerDiagnosticsRows({
    session_error: 'Failed to connect: postgresql://postgres:password123@localhost/db',
  });
  const errorRow = rows.find((r) => r.label === 'Session error');
  assert.ok(errorRow);
  assert.equal(errorRow.value, 'Failed to connect: postgresql://[redacted]:[redacted]@localhost/db');
});

test('renders graph_active with requested_mode standard', () => {
  const rows = buildAnswerDiagnosticsRows({
    graph_active: {
      enabled: false,
      requested_mode: 'standard',
      effective_enabled: false,
      reason: 'disabled_by_request',
      added_count: 0,
      added_chunks: [],
    },
  });

  const summary = rows.find((row) => row.label === 'Graph Assist');
  assert.ok(summary);
  assert.equal(summary.section, 'Graph');
  assert.ok(summary.value.includes('standard'));
  assert.ok(summary.value.includes('not effective'));
  assert.ok(summary.value.includes('disabled_by_request'));

  const requestedRow = rows.find((row) => row.label === 'Graph Assist requested');
  assert.ok(requestedRow);
  assert.equal(requestedRow.value, 'standard');
  assert.equal(requestedRow.section, 'Graph');

  const effectiveRow = rows.find((row) => row.label === 'Graph Assist effective');
  assert.ok(effectiveRow);
  assert.equal(effectiveRow.value, 'No');
});

test('renders graph_active with requested_mode graph_assist and added_chunks', () => {
  const rows = buildAnswerDiagnosticsRows({
    graph_active: {
      enabled: true,
      requested_mode: 'graph_assist',
      effective_enabled: true,
      reason: 'added',
      added_count: 1,
      added_chunks: [
        { relative_path: 'src/components/Projects.tsx', retrieval_source: 'graph_active' },
      ],
    },
  });

  const summary = rows.find((row) => row.label === 'Graph Assist');
  assert.ok(summary);
  assert.ok(summary.value.includes('graph_assist'));
  assert.ok(summary.value.includes('effective'));
  assert.ok(summary.value.includes('1 added'));

  const addedPaths = rows.find((row) => row.label === 'Graph Assist added paths');
  assert.ok(addedPaths);
  assert.deepEqual(addedPaths.value, ['src/components/Projects.tsx']);
  assert.equal(addedPaths.section, 'Graph');
});

test('handles missing graph_active safely', () => {
  const rows = buildAnswerDiagnosticsRows({
    intent: 'CODE_REQUEST',
  });
  const graphRows = rows.filter((row) => row.section === 'Graph');
  assert.equal(graphRows.length, 0);
});

test('handles null diagnostics safely for graph_active', () => {
  const rows = buildAnswerDiagnosticsRows(null);
  assert.deepEqual(rows, []);
});

test('shows disabled_by_server_config clearly when graph_assist is requested but blocked', () => {
  const rows = buildAnswerDiagnosticsRows({
    graph_active: {
      enabled: false,
      requested_mode: 'graph_assist',
      effective_enabled: false,
      reason: 'disabled',
      disabled_reason: 'disabled_by_server_config',
      added_count: 0,
      added_chunks: [],
    },
  });

  const summary = rows.find((row) => row.label === 'Graph Assist');
  assert.ok(summary);
  assert.ok(summary.value.includes('graph_assist'));
  assert.ok(summary.value.includes('not effective'));
  assert.ok(summary.value.includes('disabled_by_server_config'));

  const disabledReasonRow = rows.find((row) => row.label === 'Graph Assist disabled reason');
  assert.ok(disabledReasonRow);
  assert.equal(disabledReasonRow.value, 'disabled_by_server_config');

  const effectiveRow = rows.find((row) => row.label === 'Graph Assist effective');
  assert.ok(effectiveRow);
  assert.equal(effectiveRow.value, 'No');

  // Should NOT have added paths since count=0
  const addedPaths = rows.find((row) => row.label === 'Graph Assist added paths');
  assert.equal(addedPaths, undefined);
});
