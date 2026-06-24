/**
 * Helper utilities for classifying and grouping answer source cards.
 */

export function classifySource(source) {
  if (!source) return null;

  const file = source.file || source.relative_path || '';
  const symbol = source.symbol || source.symbol_name || '';
  const startLine = source.start_line;
  const endLine = source.end_line;
  const lines = source.lines || formatLines(startLine, endLine);
  const expansionType = source.expansion_type || source.expansionType || '';

  // Determine Badge/Kind first
  let badge = 'Code';
  const fileLower = file.toLowerCase();

  if (
    fileLower.includes('/tests/') ||
    fileLower.includes('/test_') ||
    fileLower.includes('.test.') ||
    fileLower.includes('_test.py') ||
    fileLower.includes('spec_') ||
    fileLower.endsWith('test')
  ) {
    badge = 'Test';
  } else if (
    fileLower.endsWith('.md') ||
    fileLower.endsWith('.txt') ||
    fileLower.endsWith('.rst') ||
    fileLower.includes('/docs/') ||
    fileLower.includes('/doc/')
  ) {
    badge = 'Docs';
  } else if (
    fileLower.includes('/reports/') ||
    fileLower.endsWith('_summary.json') ||
    fileLower.endsWith('_latest.json')
  ) {
    badge = 'Generated report';
  } else if (
    fileLower === 'package.json' ||
    fileLower === 'package-lock.json' ||
    fileLower === 'requirements.txt' ||
    fileLower === 'docker-compose.yml' ||
    fileLower === 'docker-compose.deploy.yml' ||
    fileLower === 'dockerfile' ||
    fileLower === 'makefile' ||
    fileLower === '.gitignore' ||
    fileLower === '.env.example' ||
    fileLower === '.env' ||
    fileLower.endsWith('.yaml') ||
    fileLower.endsWith('.yml') ||
    fileLower.endsWith('.toml') ||
    fileLower.endsWith('.ini') ||
    fileLower.endsWith('.conf')
  ) {
    badge = 'Config';
  }

  // Determine Role
  let role = 'Related';
  if (badge === 'Docs') {
    role = 'Documentation';
  } else if (badge === 'Test') {
    role = 'Tests';
  } else if (badge === 'Config') {
    role = 'Config';
  } else if (badge === 'Code') {
    if (!expansionType || expansionType === 'primary') {
      role = 'Primary implementation';
    } else {
      role = 'Supporting implementation';
    }
  }

  // Label display:
  // - If it has symbol: path :: symbol
  // - If it is doc or config: just path or clean path
  let label = file;
  if (symbol) {
    label = `${file} :: ${symbol}`;
  }

  const copyValue = [
    file,
    symbol ? `:: ${symbol}` : '',
    lines ? ` (lines ${lines})` : ''
  ].join('');

  return {
    file,
    symbol,
    lines,
    expansionType,
    badge,
    role,
    label,
    copyValue
  };
}

export function groupSources(sources) {
  const groups = {
    'Primary implementation': [],
    'Supporting implementation': [],
    'Documentation': [],
    'Tests': [],
    'Config': [],
    'Related': []
  };

  if (!Array.isArray(sources)) return groups;

  const seen = new Set();
  const uniqueSources = [];

  for (const src of sources) {
    if (!src) continue;
    const classified = classifySource(src);
    const key = `${classified.file}-${classified.symbol}-${classified.lines}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniqueSources.push({ original: src, classified });
    }
  }

  for (const item of uniqueSources) {
    groups[item.classified.role].push(item);
  }

  return groups;
}

function formatLines(startLine, endLine) {
  const start = Number(startLine);
  const end = Number(endLine);
  if (!Number.isFinite(start) || start <= 0) return '';
  if (!Number.isFinite(end) || end <= 0 || end === start) return String(start);
  return `${start}-${end}`;
}
