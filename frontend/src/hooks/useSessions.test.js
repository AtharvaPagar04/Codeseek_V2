import test from 'node:test';
import assert from 'node:assert/strict';

import {
  applyAppendMessage,
  applyClearSessionMessages,
  applyDeleteSession,
  applySetSessionThreads,
  applySetThreadMessages,
  normalizeSessionRecord,
} from './useSessions.js';


function baseSessions() {
  return [
    {
      id: 'session-1',
      repo_id: 'repo-one',
      repo_full_name: 'org/repo-one',
      created_at: '2026-06-03T00:00:00.000Z',
      last_active: '2026-06-03T00:00:00.000Z',
      active_thread_id: 'thread-a',
      threads: [
        {
          id: 'thread-a',
          messages: [{ id: 'm1', content: 'active message' }],
        },
        {
          id: 'thread-b',
          messages: [{ id: 'm2', content: 'hidden message' }],
        },
      ],
    },
  ];
}

test('clearSessionMessages only clears the active thread', () => {
  const next = applyClearSessionMessages(
    baseSessions(),
    'session-1',
    '2026-06-03T01:00:00.000Z'
  );

  const session = next[0];
  assert.equal(session.threads[0].messages.length, 0);
  assert.equal(session.threads[1].messages.length, 1);
  assert.equal(session.threads[1].messages[0].content, 'hidden message');
});

test('setThreadMessages updates only the targeted thread', () => {
  const replacement = [{ id: 'm3', content: 'replacement' }];
  const next = applySetThreadMessages(baseSessions(), 'session-1', 'thread-b', replacement);

  const session = next[0];
  assert.equal(session.threads[0].messages[0].content, 'active message');
  assert.deepEqual(session.threads[1].messages, replacement);
});

test('appendMessage appends to the targeted thread without leaking to hidden threads', () => {
  const next = applyAppendMessage(
    baseSessions(),
    'session-1',
    'thread-a',
    { id: 'm4', content: 'new active message' },
    '2026-06-03T01:00:00.000Z'
  );

  const session = next[0];
  assert.equal(session.threads[0].messages.length, 2);
  assert.equal(session.threads[0].messages[1].content, 'new active message');
  assert.equal(session.threads[1].messages.length, 1);
  assert.equal(session.threads[1].messages[0].content, 'hidden message');
});

test('appendMessage replacement only replaces inside the targeted thread', () => {
  const sessions = [
    {
      ...baseSessions()[0],
      threads: [
        {
          id: 'thread-a',
          messages: [{ id: 'replace-me', content: 'loading', loading: true }],
        },
        {
          id: 'thread-b',
          messages: [{ id: 'replace-me', content: 'hidden loading', loading: true }],
        },
      ],
    },
  ];

  const next = applyAppendMessage(
    sessions,
    'session-1',
    'thread-a',
    { __replaceId: 'replace-me', id: 'replace-me', content: 'done', loading: false },
    '2026-06-03T01:00:00.000Z'
  );

  const session = next[0];
  assert.equal(session.threads[0].messages[0].content, 'done');
  assert.equal(session.threads[1].messages[0].content, 'hidden loading');
});

test('setSessionThreads preserves active thread when still present', () => {
  const next = applySetSessionThreads(baseSessions(), 'session-1', [
    { id: 'thread-b', title: 'Hidden Thread' },
    { id: 'thread-a', title: 'Active Thread' },
  ]);

  assert.equal(next[0].active_thread_id, 'thread-a');
  assert.deepEqual(next[0].threads[0].messages, []);
  assert.deepEqual(next[0].threads[1].messages, []);
});

test('setSessionThreads falls back to first visible thread when active thread disappears', () => {
  const next = applySetSessionThreads(baseSessions(), 'session-1', [
    { id: 'thread-c', title: 'Only Thread Left' },
  ]);

  assert.equal(next[0].active_thread_id, 'thread-c');
  assert.equal(next[0].threads.length, 1);
});

test('normalizeSessionRecord preserves local thread state when backend reuses a session', () => {
  const next = normalizeSessionRecord(
    {
      id: 'session-1',
      repo_full_name: 'org/repo-one',
      status: 'ready',
      error: '',
      created_at: '2026-06-03T00:00:00.000Z',
    },
    baseSessions()[0],
    {
      now: '2026-06-03T02:00:00.000Z',
      lastActive: '2026-06-03T02:00:00.000Z',
    }
  );

  assert.equal(next.active_thread_id, 'thread-a');
  assert.equal(next.threads.length, 2);
  assert.equal(next.threads[0].messages[0].content, 'active message');
  assert.equal(next.last_active, '2026-06-03T02:00:00.000Z');
});



test('applyDeleteSession removes the target session', () => {
  const sessions = [
    { id: 'session-1', repo_id: 'repo-one', created_at: '2026-06-01T00:00:00Z', last_active: '2026-06-01T00:00:00Z' },
    { id: 'session-2', repo_id: 'repo-two', created_at: '2026-06-02T00:00:00Z', last_active: '2026-06-02T00:00:00Z' },
  ];
  const next = applyDeleteSession(sessions, 'session-1');
  assert.equal(next.length, 1);
  assert.equal(next[0].id, 'session-2');
});

test('applyDeleteSession is a no-op for unknown session id', () => {
  const sessions = [
    { id: 'session-1', repo_id: 'repo-one', created_at: '2026-06-01T00:00:00Z', last_active: '2026-06-01T00:00:00Z' },
  ];
  const next = applyDeleteSession(sessions, 'nonexistent');
  assert.equal(next.length, 1);
  assert.equal(next[0].id, 'session-1');
});

test('applyDeleteSession does not affect unrelated sessions', () => {
  const sessions = [
    { id: 'a', repo_id: 'ra', created_at: '2026-06-01T00:00:00Z', last_active: '2026-06-01T00:00:00Z' },
    { id: 'b', repo_id: 'rb', created_at: '2026-06-02T00:00:00Z', last_active: '2026-06-02T00:00:00Z' },
    { id: 'c', repo_id: 'rc', created_at: '2026-06-03T00:00:00Z', last_active: '2026-06-03T00:00:00Z' },
  ];
  const next = applyDeleteSession(sessions, 'b');
  assert.equal(next.length, 2);
  assert.ok(next.every((s) => s.id !== 'b'));
  assert.ok(next.some((s) => s.id === 'a'));
  assert.ok(next.some((s) => s.id === 'c'));
});
