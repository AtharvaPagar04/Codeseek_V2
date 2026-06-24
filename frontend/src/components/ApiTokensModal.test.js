import test from 'node:test';
import assert from 'node:assert/strict';
import { validateEmbeddingDimensions } from '../utils/validation.js';

test('validateEmbeddingDimensions', async (t) => {
  await t.test('allows local mode any dimensions', () => {
    assert.equal(validateEmbeddingDimensions('local', 'openai/text-embedding-3-large', '384'), null);
  });

  await t.test('blocks 384 for openai/text-embedding-3-large in api mode', () => {
    const err = validateEmbeddingDimensions('api', 'openai/text-embedding-3-large', '384');
    assert.equal(err, '384 is not valid for openai/text-embedding-3-large. Use Auto, 256, 1024, or 3072.');
  });

  await t.test('allows 0 (Auto) for openai/text-embedding-3-large in api mode', () => {
    const err = validateEmbeddingDimensions('api', 'openai/text-embedding-3-large', '0');
    assert.equal(err, null);
  });

  await t.test('allows 256 for openai/text-embedding-3-large in api mode', () => {
    const err = validateEmbeddingDimensions('api', 'openai/text-embedding-3-large', '256');
    assert.equal(err, null);
  });

  await t.test('allows unknown model to have any dimension in api mode', () => {
    const err = validateEmbeddingDimensions('api', 'some/unknown-model', '1234');
    assert.equal(err, null);
  });
});
