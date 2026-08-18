import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import { useChat } from './useChat.js';


const event = (type, requestId, fields = {}) => JSON.stringify({
  type,
  protocol_version: 2,
  request_id: requestId,
  ...fields,
});

async function runHook(makeRecords) {
  const originalFetch = globalThis.fetch;
  const appended = [];
  let hook;
  globalThis.localStorage = { getItem: () => null };
  globalThis.fetch = async (_url, options) => {
    const requestId = options.headers['X-Request-Id'];
    const bytes = new TextEncoder().encode(makeRecords(requestId));
    let read = false;
    return {
      ok: true,
      body: {
        getReader: () => ({
          read: async () => read
            ? { value: undefined, done: true }
            : ((read = true), { value: bytes, done: false }),
        }),
      },
    };
  };

  try {
    function Harness() {
      hook = useChat({ appendMessage: (_sessionId, _threadId, message) => appended.push(message) });
      return null;
    }
    renderToStaticMarkup(React.createElement(Harness));
    await hook.sendMessage(
      { id: 'session', active_thread_id: 'thread', status: 'ready' },
      'question',
    );
    return appended;
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
}

test('useChat replaces provisional deltas with the persisted authoritative answer', async () => {
  const messages = await runHook((requestId) => [
    event('delta', requestId, { text: 'The answer is in old_file.py.' }),
    event('final_answer', requestId, {
      message_id: 'persisted-message',
      text: 'The answer is in corrected_file.py.',
      generation_status: 'complete',
      authoritative: true,
    }),
    event('sources', requestId, { message_id: 'persisted-message', sources: [] }),
    event('done', requestId, { message_id: 'persisted-message', status: 'complete' }),
  ].join('\n'));

  const final = messages.at(-1);
  assert.equal(final.id, 'persisted-message');
  assert.equal(final.content, 'The answer is in corrected_file.py.');
  assert.equal(final.loading, false);
  assert.equal(final.error, false);
  assert.equal(final.content.includes('old_file.py'), false);
});

test('useChat clears loading for partial, error, and incomplete terminal states', async (t) => {
  await t.test('partial', async () => {
    const messages = await runHook((requestId) => [
      event('delta', requestId, { text: 'provisional' }),
      event('final_answer', requestId, {
        message_id: 'partial-message', text: 'saved partial', generation_status: 'partial', authoritative: true,
      }),
      event('sources', requestId, { message_id: 'partial-message', sources: [] }),
      event('done', requestId, { message_id: 'partial-message', status: 'partial' }),
    ].join('\n'));
    assert.deepEqual(
      { content: messages.at(-1).content, loading: messages.at(-1).loading, status: messages.at(-1).generation_status },
      { content: 'saved partial', loading: false, status: 'partial' },
    );
  });

  await t.test('error after provisional text', async () => {
    const messages = await runHook((requestId) => [
      event('delta', requestId, { text: 'provisional' }),
      event('error', requestId, { code: 'generation_failed', message: 'Generation failed.', retryable: false }),
      event('done', requestId, { message_id: null, status: 'error' }),
    ].join('\n'));
    assert.deepEqual(
      { content: messages.at(-1).content, loading: messages.at(-1).loading, error: messages.at(-1).error },
      { content: 'Generation failed.', loading: false, error: true },
    );
  });

  await t.test('authoritative answer without done', async () => {
    const messages = await runHook((requestId) => event('final_answer', requestId, {
      message_id: 'saved-message', text: 'saved answer', generation_status: 'complete', authoritative: true,
    }));
    assert.deepEqual(
      { content: messages.at(-1).content, loading: messages.at(-1).loading, error: messages.at(-1).error },
      { content: 'saved answer', loading: false, error: true },
    );
  });
});
