const MAX_SOURCE_ITEMS = 6;

export function sanitizeCredentialsInString(text) {
  if (typeof text !== 'string') {
    return text ? String(text) : '';
  }
  let result = text.replace(/bearer\s+[a-z0-9_\-\.]+/gi, 'Bearer [redacted]');
  result = result.replace(/([a-z0-9+.-]+:\/\/)([^@/]+)(@)/gi, (match, scheme, creds, at) => {
    if (creds.includes(':')) {
      return `${scheme}[redacted]:[redacted]${at}`;
    }
    return `${scheme}[redacted]${at}`;
  });
  return result;
}

function safeString(value) {
  const str = typeof value === 'string' ? value.trim() : `${value ?? ''}`.trim();
  return sanitizeCredentialsInString(str);
}

function formatLineRange(source) {
  const start = Number(source?.start_line);
  const end = Number(source?.end_line);
  if (!Number.isFinite(start) || start <= 0) return '';
  if (!Number.isFinite(end) || end <= 0 || end === start) return `L${start}`;
  return `L${start}–${end}`;
}

export function summarizeDiagnosticSource(source) {
  if (!source || typeof source !== 'object') return '';

  const relativePath = safeString(source.relative_path || source.file);
  if (!relativePath) return '';

  const symbolName = safeString(source.symbol_name || source.symbol);
  const lines = formatLineRange(source);
  const parts = [relativePath];
  if (symbolName) parts.push(`:: ${symbolName}`);
  if (lines) parts.push(`(${lines})`);
  return parts.join(' ');
}

function summarizeEvidenceConfidence(confidence) {
  if (!confidence || typeof confidence !== 'object') return '';
  const parts = [];
  const level = safeString(confidence.level);
  const count = Number(confidence.count);
  const reason = safeString(confidence.reason);
  if (level) parts.push(level);
  if (Number.isFinite(count) && count >= 0) parts.push(`${count} hit${count === 1 ? '' : 's'}`);
  if (reason) parts.push(reason);
  return parts.join(' · ');
}

function summarizeSourceFilter(sourceFilter) {
  if (!sourceFilter || typeof sourceFilter !== 'object') return '';
  const parts = [];
  const selected = Number(sourceFilter.selected_primary);
  const expanded = Number(sourceFilter.selected_expanded);
  const display = Number(sourceFilter.display_count);
  const reasoning = Number(sourceFilter.reasoning_count);
  if (Number.isFinite(selected)) parts.push(`primary ${selected}`);
  if (Number.isFinite(expanded)) parts.push(`expanded ${expanded}`);
  if (Number.isFinite(display)) parts.push(`display ${display}`);
  if (Number.isFinite(reasoning)) parts.push(`reasoning ${reasoning}`);
  return parts.join(' · ');
}

function summarizeBoolean(value) {
  if (typeof value !== 'boolean') return '';
  return value ? 'Yes' : 'No';
}

function summarizeNumber(value, digits = 3) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '';
  return `${num.toFixed(digits)}`.replace(/\.?0+$/, '');
}

function compactSourceList(items) {
  if (!Array.isArray(items) || items.length === 0) return [];
  const seen = new Set();
  const compacted = [];
  for (const item of items) {
    const summary = typeof item === 'string'
      ? safeString(item)
      : summarizeDiagnosticSource(item);
    if (!summary || seen.has(summary)) continue;
    seen.add(summary);
    compacted.push(sanitizeCredentialsInString(summary));
    if (compacted.length >= MAX_SOURCE_ITEMS) break;
  }
  return compacted;
}

export function buildAnswerDiagnosticsRows(diagnostics) {
  if (!diagnostics || typeof diagnostics !== 'object') return [];

  const rows = [];
  const addTextRow = (label, value, section = 'General', isAdvanced = false) => {
    const text = safeString(value);
    if (!text) return;
    rows.push({ label, kind: 'text', value: text, section, isAdvanced });
  };
  const addListRow = (label, items, section = 'General', isAdvanced = false) => {
    const values = compactSourceList(items);
    if (values.length === 0) return;
    rows.push({ label, kind: 'list', value: values, section, isAdvanced });
  };

  addTextRow('Intent', diagnostics.intent, 'Intent', false);
  addTextRow('Primary intent', diagnostics.primary_intent, 'Intent', true);
  addTextRow('Response mode', diagnostics.response_mode, 'Intent', true);
  addTextRow(
    'Model',
    [diagnostics.provider, diagnostics.model].filter(Boolean).join(' / '),
    'Model',
    false
  );
  addTextRow('Routing mode', diagnostics.routing_mode, 'Intent', true);

  addTextRow('Evidence confidence', summarizeEvidenceConfidence(diagnostics.evidence_confidence), 'Sources', false);
  addTextRow('Source filter', summarizeSourceFilter(diagnostics.source_filter), 'Sources', true);

  addListRow('Selected sources', diagnostics.selected_sources, 'Sources', true);
  addListRow('Reasoning sources', diagnostics.reasoning_sources, 'Sources', true);
  addListRow('Rendered sources', diagnostics.rendered_sources, 'Sources', true);
  addTextRow('Follow-up', summarizeBoolean(diagnostics.memory?.is_followup), 'Intent', false);
  addTextRow('Topic shift detected', summarizeBoolean(diagnostics.memory?.topic_shift_detected), 'Intent', true);
  addTextRow('Follow-up confidence', summarizeNumber(diagnostics.memory?.followup_confidence), 'Intent', true);
  addTextRow('Query similarity', summarizeNumber(diagnostics.memory?.query_similarity), 'Intent', true);
  addTextRow('Keyword overlap', summarizeNumber(diagnostics.memory?.keyword_overlap), 'Intent', true);
  addTextRow('Similarity method', diagnostics.memory?.similarity_method, 'Intent', true);
  addTextRow('Valid referent', summarizeBoolean(diagnostics.memory?.has_valid_referent), 'Intent', true);
  addTextRow('History injected', summarizeBoolean(diagnostics.memory?.history_injected), 'Sources', false);
  addTextRow('History turns used', diagnostics.memory?.history_turns_used, 'Sources', true);
  addTextRow('Query rewritten', summarizeBoolean(diagnostics.rewrite?.query_rewritten), 'Intent', false);
  addTextRow('Rewrite mode', diagnostics.rewrite?.rewrite_mode, 'Intent', true);
  addTextRow('Rewrite anchor', diagnostics.rewrite?.rewrite_anchor, 'Intent', true);
  addTextRow('Previous candidates injected', diagnostics.retrieval?.previous_candidates_injected, 'Sources', false);
  addTextRow('Retrieval confidence', diagnostics.retrieval?.retrieval_confidence, 'Sources', false);
  addTextRow('Exact retrieval hit', summarizeBoolean(diagnostics.retrieval?.exact_hit), 'Sources', true);
  addTextRow('Multi-layer hit', summarizeBoolean(diagnostics.retrieval?.multi_layer_hit), 'Sources', true);
  addTextRow('Top score', summarizeNumber(diagnostics.retrieval?.top_score), 'Sources', true);
  addTextRow('Candidate count', diagnostics.retrieval?.candidate_count, 'Sources', true);
  addTextRow('Low-confidence gate', summarizeBoolean(diagnostics.retrieval?.low_confidence_gate), 'Sources', true);
  addListRow('Strong new entities', diagnostics.retrieval?.strong_new_entities, 'Sources', true);


  return rows;
}
