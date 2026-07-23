import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_API_EMBEDDING_MODEL,
  DEFAULT_LOCAL_EMBEDDING_MODEL,
  defaultEmbeddingModelForMode,
} from '../utils/validation.js';

test('defaultEmbeddingModelForMode', async (t) => {
  await t.test('uses text-embedding-3-small for API mode', () => {
    assert.equal(defaultEmbeddingModelForMode('api'), DEFAULT_API_EMBEDDING_MODEL);
    assert.equal(DEFAULT_API_EMBEDDING_MODEL, 'text-embedding-3-small');
  });

  await t.test('uses nomic embed text for local mode', () => {
    assert.equal(defaultEmbeddingModelForMode('local'), DEFAULT_LOCAL_EMBEDDING_MODEL);
    assert.equal(DEFAULT_LOCAL_EMBEDDING_MODEL, 'nomic-embed-text:latest');
  });

  await t.test('falls back to API default for unknown modes', () => {
    assert.equal(defaultEmbeddingModelForMode(''), DEFAULT_API_EMBEDDING_MODEL);
  });
});
