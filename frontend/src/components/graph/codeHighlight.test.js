import test from 'node:test';
import assert from 'node:assert/strict';

import {
  languageLabel,
  lineCount,
  normalizeLanguage,
  tokenizeCodeLine,
  tokenClass,
} from './codeHighlight.js';

test('language helpers prefer specific file extensions for TSX and JSX', () => {
  assert.equal(normalizeLanguage('typescript', 'src/components/Projects.tsx'), 'tsx');
  assert.equal(languageLabel('typescript', 'src/components/Projects.tsx'), 'TypeScript TSX');
  assert.equal(normalizeLanguage('javascript', 'src/components/App.jsx'), 'jsx');
  assert.equal(languageLabel('', 'scripts/build.py'), 'Python');
  assert.equal(languageLabel('', 'README.md'), 'Text');
});

test('tokenizeCodeLine highlights keywords, function names, strings, and brackets', () => {
  const tokens = tokenizeCodeLine('export function Projects() { return "CodeSeek"; }');
  const tokenTypes = tokens.map((token) => token.type);

  assert.ok(tokenTypes.includes('keyword'));
  assert.ok(tokenTypes.includes('function'));
  assert.ok(tokenTypes.includes('string'));
  assert.ok(tokenTypes.includes('bracket'));
});

test('tokenizeCodeLine highlights JSX tags and attributes', () => {
  const tokens = tokenizeCodeLine('<ProjectCard title="CodeSeek" isActive={true} />');
  const typed = tokens.filter((token) => token.type !== 'plain');

  assert.ok(typed.some((token) => token.type === 'tag' && token.text.includes('ProjectCard')));
  assert.ok(typed.some((token) => token.type === 'attribute' && token.text === 'title'));
  assert.ok(typed.some((token) => token.type === 'string' && token.text.includes('CodeSeek')));
  assert.ok(typed.some((token) => token.type === 'keyword' && token.text === 'true'));
});

test('token classes map highlighted token kinds to visible dark-theme colors', () => {
  assert.match(tokenClass('keyword'), /fuchsia/);
  assert.match(tokenClass('string'), /emerald/);
  assert.match(tokenClass('function'), /violet/);
  assert.match(tokenClass('tag'), /cyan/);
  assert.match(tokenClass('attribute'), /amber/);
  assert.match(tokenClass('plain'), /slate/);
});

test('lineCount handles empty and multiline snippets', () => {
  assert.equal(lineCount(''), 0);
  assert.equal(lineCount('const a = 1;'), 1);
  assert.equal(lineCount('const a = 1;\nconst b = 2;'), 2);
});
