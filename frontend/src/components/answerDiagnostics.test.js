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
  const renderedRow = rows.find((row) => row.label === 'Rendered sources');
  assert.equal(renderedRow.value[0], 'backend/evals/run_safe_evals.py :: main (L10–48)');
  const selectedRow = rows.find((row) => row.label === 'Selected sources');
  assert.equal(selectedRow.value[0], 'backend/evals/run_safe_evals.py :: main (L10–48)');
  const strongEntitiesRow = rows.find((row) => row.label === 'Strong new entities');
  assert.deepEqual(strongEntitiesRow.value, ['backend/retrieval/api_service.py', '_require_auth']);
  assert.ok(rows.every((row) => JSON.stringify(row).indexOf('secret') === -1));
  assert.ok(rows.every((row) => JSON.stringify(row).indexOf('hidden') === -1));
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
