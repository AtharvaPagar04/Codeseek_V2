const KEYWORDS = new Set([
  'async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'default',
  'do', 'else', 'export', 'extends', 'false', 'finally', 'for', 'from', 'function',
  'if', 'import', 'in', 'interface', 'let', 'new', 'null', 'of', 'return', 'switch',
  'throw', 'true', 'try', 'type', 'undefined', 'var', 'while',
]);

const LANGUAGE_LABELS = {
  js: 'JavaScript',
  jsx: 'JavaScript JSX',
  javascript: 'JavaScript',
  ts: 'TypeScript',
  tsx: 'TypeScript TSX',
  typescript: 'TypeScript',
  py: 'Python',
  python: 'Python',
  json: 'JSON',
  text: 'Text',
};

export function normalizeLanguage(language = '', path = '') {
  const raw = String(language || '').trim().toLowerCase();
  const extension = String(path || '').split('.').pop()?.toLowerCase() || '';
  if ((raw === 'typescript' || raw === 'ts') && extension === 'tsx') return 'tsx';
  if ((raw === 'javascript' || raw === 'js') && extension === 'jsx') return 'jsx';
  if (raw) {
    if (raw === 'typescript') return 'ts';
    if (raw === 'javascript') return 'js';
    return raw;
  }
  return LANGUAGE_LABELS[extension] ? extension : 'text';
}

export function languageLabel(language = '', path = '') {
  const normalized = normalizeLanguage(language, path);
  return LANGUAGE_LABELS[normalized] || normalized.toUpperCase() || 'Text';
}

export function tokenClass(type) {
  switch (type) {
    case 'keyword':
      return 'text-fuchsia-300';
    case 'string':
      return 'text-emerald-300';
    case 'function':
      return 'text-violet-300';
    case 'tag':
      return 'text-cyan-300';
    case 'attribute':
      return 'text-amber-300';
    case 'bracket':
      return 'text-sky-300';
    case 'comment':
      return 'text-slate-500 italic';
    case 'number':
      return 'text-orange-300';
    default:
      return 'text-slate-100';
  }
}

export function tokenizeCodeLine(line = '') {
  const source = String(line);
  const pattern = /(\/\/.*$|\/\*.*?\*\/|`(?:\\.|[^`])*`|'(?:\\.|[^'])*'|"(?:\\.|[^"])*"|<\/?[A-Za-z][\w.:-]*|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|[{}[\]();,.<>/=:+\-*?!|&])/g;
  const tokens = [];
  let cursor = 0;
  let match;

  while ((match = pattern.exec(source)) !== null) {
    if (match.index > cursor) {
      tokens.push({ text: source.slice(cursor, match.index), type: 'plain' });
    }

    const text = match[0];
    const next = source.slice(pattern.lastIndex);
    let type = 'plain';

    if (text.startsWith('//') || text.startsWith('/*')) {
      type = 'comment';
    } else if (text.startsWith('"') || text.startsWith("'") || text.startsWith('`')) {
      type = 'string';
    } else if (/^<\/?[A-Za-z]/.test(text)) {
      type = 'tag';
    } else if (/^\d/.test(text)) {
      type = 'number';
    } else if (KEYWORDS.has(text)) {
      type = 'keyword';
    } else if (/^[A-Za-z_$]/.test(text) && /^\s*=/.test(next)) {
      type = 'attribute';
    } else if (/^[A-Za-z_$]/.test(text) && /^\s*\(/.test(next)) {
      type = 'function';
    } else if (/^[{}[\]();,.<>/=:+\-*?!|&]$/.test(text)) {
      type = 'bracket';
    }

    tokens.push({ text, type });
    cursor = pattern.lastIndex;
  }

  if (cursor < source.length) {
    tokens.push({ text: source.slice(cursor), type: 'plain' });
  }
  return tokens.length ? tokens : [{ text: '', type: 'plain' }];
}

export function lineCount(code = '') {
  if (!code) return 0;
  return String(code).split('\n').length;
}
