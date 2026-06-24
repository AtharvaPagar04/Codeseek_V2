import test from 'node:test';
import assert from 'node:assert/strict';

import { formatApiError } from './api.js';
import { getBackendApiKey } from './storage.js';

test('formatApiError maps provider auth failures to actionable copy', () => {
  const message = formatApiError({
    action: 'Query',
    status: 400,
    detail: 'Provider API key rejected or lacks permission.',
  });

  assert.match(message, /provider rejected/i);
  assert.match(message, /update the provider configuration/i);
});

test('formatApiError maps unsupported provider configuration copy', () => {
  const message = formatApiError({
    action: 'Query',
    status: 400,
    detail: 'Unsupported LLM provider configuration: mystery',
  });

  assert.match(message, /provider configuration is invalid/i);
});

test('formatApiError maps rate-limit copy', () => {
  const message = formatApiError({
    action: 'Query',
    status: 429,
    detail: 'Provider rate limit reached. Wait and retry, or switch provider credentials.',
  });

  assert.match(message, /rate limit reached/i);
  assert.match(message, /switch provider credentials/i);
});

test('fetchLatestEvaluationReport invokes the correct endpoint', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({ status: 'PASS', available: true })
    };
  };

  try {
    const { fetchLatestEvaluationReport } = await import('./api.js');
    const report = await fetchLatestEvaluationReport('session-123');
    assert.equal(report.status, 'PASS');
    assert.equal(report.available, true);
    assert.match(calledUrl, /\/api\/v1\/sessions\/session-123\/evaluation\/latest/);
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('fetchLatestGlobalEvaluationReport invokes the correct endpoint', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({ status: 'PASS', available: true })
    };
  };

  try {
    const { fetchLatestGlobalEvaluationReport } = await import('./api.js');
    const report = await fetchLatestGlobalEvaluationReport();
    assert.equal(report.status, 'PASS');
    assert.equal(report.available, true);
    assert.match(calledUrl, /\/api\/v1\/evals\/latest/);
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('querySession invokes the correct endpoint and includes credentials/auth headers', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        answer: 'Query response',
        sources: [],
        diagnostics: { response_mode: 'code_snippet' },
      })
    };
  };

  try {
    const { querySession } = await import('./api.js');
    const result = await querySession({ question: 'How to use this?', session_id: 'session-123' });
    assert.equal(result.answer, 'Query response');
    assert.equal(result.diagnostics.response_mode, 'code_snippet');
    assert.match(calledUrl, /\/api\/v1\/query/);
    assert.equal(calledOptions.credentials, 'include');
    assert.equal(calledOptions.headers['Content-Type'], 'application/json');
    assert.equal(calledOptions.headers['Authorization'], undefined);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('indexLatestSession invokes the correct endpoint and includes headers', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        session_id: 'session-123',
        status: 'indexing',
        message: 'Indexing latest repository state started.',
        freshness_status: 'indexing',
      })
    };
  };

  try {
    const { indexLatestSession } = await import('./api.js');
    const result = await indexLatestSession('session-123');
    assert.equal(result.status, 'indexing');
    assert.equal(result.freshness_status, 'indexing');
    assert.match(calledUrl, /\/api\/v1\/sessions\/session-123\/index-latest/);
    assert.equal(calledOptions.method, 'POST');
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('fetchIndexPreview invokes the correct endpoint and includes credentials/auth headers', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        session_id: 'session-123',
        freshness_status: 'dirty_worktree',
        worktree_dirty: true,
        modified_files_count: 2,
      })
    };
  };

  try {
    const { fetchIndexPreview } = await import('./api.js');
    const result = await fetchIndexPreview('session-123');
    assert.equal(result.freshness_status, 'dirty_worktree');
    assert.equal(result.worktree_dirty, true);
    assert.equal(result.modified_files_count, 2);
    assert.match(calledUrl, /\/api\/v1\/sessions\/session-123\/index-preview/);
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('indexSessionIncremental invokes the correct endpoint and includes headers', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        session_id: 'session-123',
        status: 'indexing',
        message: 'Incremental indexing started.',
        freshness_status: 'indexing',
      })
    };
  };

  try {
    const { indexSessionIncremental } = await import('./api.js');
    const result = await indexSessionIncremental('session-123');
    assert.equal(result.status, 'indexing');
    assert.match(calledUrl, /\/api\/v1\/sessions\/session-123\/index-incremental/);
    assert.equal(calledOptions.method, 'POST');
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('formatApiError handles incremental indexing errors cleanly', () => {
  // Feature disabled
  let msg = formatApiError({
    action: 'Incremental indexing',
    status: 403,
    detail: 'Incremental reindexing is disabled.',
  });
  assert.equal(msg, 'Incremental indexing is not enabled on this server.');

  // Plan unavailable
  msg = formatApiError({
    action: 'Incremental indexing',
    status: 400,
    detail: 'plan unavailable: Missing index metadata',
  });
  assert.equal(msg, 'Incremental preview is unavailable. Use Index latest instead.');

  // Active indexing
  msg = formatApiError({
    action: 'Incremental indexing',
    status: 400,
    detail: 'already in progress',
  });
  assert.equal(msg, 'Indexing is already running.');

  // Generic failure
  msg = formatApiError({
    action: 'Incremental indexing',
    status: 500,
    detail: 'Internal server error',
  });
  assert.equal(msg, 'Incremental indexing failed to start. Use Index latest as a fallback.');
});


test('fetchLatestIndexingJob invokes the correct endpoint and includes credentials/auth headers', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  const mockJob = {
    session_id: 'session-abc',
    job_id: 'job-001',
    indexing_mode: 'incremental',
    status: 'succeeded',
    current_stage: 'done',
    files_indexed: 5,
    chunks_generated: 80,
    embeddings_stored: 80,
    started_at: '2026-06-12T10:00:00Z',
    updated_at: '2026-06-12T10:01:00Z',
    completed_at: '2026-06-12T10:01:00Z',
    error: null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => mockJob,
    };
  };

  try {
    const { fetchLatestIndexingJob } = await import('./api.js');
    const result = await fetchLatestIndexingJob('session-abc');
    assert.equal(result.job_id, 'job-001');
    assert.equal(result.indexing_mode, 'incremental');
    assert.equal(result.status, 'succeeded');
    assert.match(calledUrl, /\/api\/v1\/sessions\/session-abc\/indexing-job\/latest/);
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('fetchLatestIndexingJob handles response with latest_job null', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  const mockResponse = {
    session_id: 'session-abc',
    latest_job: null,
  };

  globalThis.fetch = async (url, options) => {
    return {
      ok: true,
      json: async () => mockResponse,
    };
  };

  try {
    const { fetchLatestIndexingJob } = await import('./api.js');
    const result = await fetchLatestIndexingJob('session-abc');
    assert.equal(result.session_id, 'session-abc');
    assert.equal(result.latest_job, null);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('cancelLatestIndexingJob calls correct endpoint with POST and includes credentials', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        session_id: 'session-xyz',
        job_id: 'job-002',
        status: 'cancelling',
        message: 'Cancellation requested.',
      }),
    };
  };

  try {
    const { cancelLatestIndexingJob } = await import('./api.js');
    const result = await cancelLatestIndexingJob('session-xyz');
    assert.equal(result.status, 'cancelling');
    assert.equal(result.job_id, 'job-002');
    assert.match(calledUrl, /\/api\/v1\/sessions\/session-xyz\/indexing-job\/cancel/);
    assert.equal(calledOptions.method, 'POST');
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('cancelLatestIndexingJob handles no_active_job response gracefully', async () => {
  const originalFetch = globalThis.fetch;

  globalThis.localStorage = {
    getItem: (key) => null,
    setItem: () => null,
    removeItem: () => null,
  };

  globalThis.fetch = async (url, options) => ({
    ok: true,
    json: async () => ({
      session_id: 'session-xyz',
      job_id: null,
      status: 'no_active_job',
      message: 'No active indexing job found for this session.',
    }),
  });

  try {
    const { cancelLatestIndexingJob } = await import('./api.js');
    const result = await cancelLatestIndexingJob('session-xyz');
    assert.equal(result.status, 'no_active_job');
    assert.equal(result.job_id, null);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('fetchIndexingJobHistory calls correct endpoint and returns jobs list', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => { },
    removeItem: () => { },
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        session_id: 'sess-history-001',
        jobs: [
          {
            job_id: 'job-aaa',
            indexing_mode: 'incremental',
            status: 'cancelled',
            current_stage: 'embedding',
            files_indexed: 3,
            chunks_generated: 42,
            embeddings_stored: 30,
            cancel_requested: true,
            started_at: '2026-06-12T10:00:00Z',
            updated_at: '2026-06-12T10:05:00Z',
            completed_at: '2026-06-12T10:05:01Z',
            error: 'Indexing cancelled by user.',
          },
        ],
      }),
    };
  };

  try {
    const { fetchIndexingJobHistory } = await import('./api.js');
    const result = await fetchIndexingJobHistory('sess-history-001');
    assert.equal(result.session_id, 'sess-history-001');
    assert.ok(Array.isArray(result.jobs));
    assert.equal(result.jobs.length, 1);
    assert.equal(result.jobs[0].job_id, 'job-aaa');
    assert.equal(result.jobs[0].status, 'cancelled');
    assert.match(calledUrl, /\/api\/v1\/sessions\/sess-history-001\/indexing-jobs/);
    assert.equal(calledOptions.credentials, 'include');
    // Default limit=20 is passed as query param
    assert.match(calledUrl, /limit=20/);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('fetchIndexingJobHistory supports custom limit param', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => { },
    removeItem: () => { },
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    return {
      ok: true,
      json: async () => ({ session_id: 'sess-lim', jobs: [] }),
    };
  };

  try {
    const { fetchIndexingJobHistory } = await import('./api.js');
    const result = await fetchIndexingJobHistory('sess-lim', 5);
    assert.equal(result.jobs.length, 0);
    assert.match(calledUrl, /limit=5/);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('deleteSessionApi calls correct endpoint with DELETE method and returns structured result', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => { },
    removeItem: () => { },
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({
        deleted: true,
        session_id: 'sess-del-001',
        qdrant_collection_deleted: true,
        warnings: [],
      }),
    };
  };

  try {
    const { deleteSessionApi } = await import('./api.js');
    const result = await deleteSessionApi('sess-del-001');
    assert.equal(result.deleted, true);
    assert.equal(result.session_id, 'sess-del-001');
    assert.equal(result.qdrant_collection_deleted, true);
    assert.deepEqual(result.warnings, []);
    assert.match(calledUrl, /\/api\/v1\/sessions\/sess-del-001/);
    assert.equal(calledOptions.method, 'DELETE');
    assert.equal(calledOptions.credentials, 'include');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('deleteSessionApi returns warnings when qdrant cleanup fails', async () => {
  const originalFetch = globalThis.fetch;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => { },
    removeItem: () => { },
  };

  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      deleted: true,
      session_id: 'sess-del-002',
      qdrant_collection_deleted: false,
      warnings: ['Qdrant collection could not be deleted: connection refused.'],
    }),
  });

  try {
    const { deleteSessionApi } = await import('./api.js');
    const result = await deleteSessionApi('sess-del-002');
    assert.equal(result.deleted, true);
    assert.equal(result.qdrant_collection_deleted, false);
    assert.equal(result.warnings.length, 1);
    assert.match(result.warnings[0], /qdrant/i);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});


test('deleteSessionApi propagates 409 active-indexing error', async () => {
  const originalFetch = globalThis.fetch;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => { },
    removeItem: () => { },
  };

  globalThis.fetch = async () => ({
    ok: false,
    status: 409,
    json: async () => ({
      detail: 'Cannot delete a session that is actively indexing.',
    }),
  });

  try {
    const { deleteSessionApi } = await import('./api.js');
    let threw = false;
    try {
      await deleteSessionApi('sess-del-003');
    } catch (err) {
      threw = true;
      assert.ok(err.message || String(err));
    }
    assert.ok(threw, 'Expected deleteSessionApi to throw on 409');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('querySessionStream parses NDJSON chunks and invokes callbacks', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let calledOptions = null;

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => null,
    removeItem: () => null,
  };

  const chunks = [
    '{"type": "status", "message": "Retrieving..."}\n',
    '{"type": "delta", "text": "Hello"}\n',
    '{"type": "delta", "text": " world"}\n',
    '{"type": "sources", "sources": [], "context_tokens": 123, "evidence_confidence": "strong", "diagnostics": {"response_mode": "code_snippet", "memory": {"history_injected": false}}}\n',
    '{"type": "done"}\n'
  ];

  let chunkIdx = 0;
  const mockReader = {
    read: async () => {
      if (chunkIdx < chunks.length) {
        const encoder = new TextEncoder();
        const value = encoder.encode(chunks[chunkIdx++]);
        return { value, done: false };
      }
      return { value: undefined, done: true };
    }
  };

  const mockBody = {
    getReader: () => mockReader
  };

  globalThis.fetch = async (url, options) => {
    calledUrl = url;
    calledOptions = options;
    return {
      ok: true,
      body: mockBody
    };
  };

  try {
    const { querySessionStream } = await import('./api.js');
    const statuses = [];
    const deltas = [];
    let receivedSources = null;
    let doneCalled = false;

    await querySessionStream({
      question: 'Hello?',
      session_id: 'session-123',
      onStatus: (msg) => statuses.push(msg),
      onDelta: (text) => deltas.push(text),
      onSources: (data) => { receivedSources = data; },
      onDone: () => { doneCalled = true; }
    });

    assert.deepEqual(statuses, ['Retrieving...']);
    assert.deepEqual(deltas, ['Hello', ' world']);
    assert.ok(receivedSources);
    assert.equal(receivedSources.context_tokens, 123);
    assert.equal(receivedSources.evidence_confidence, 'strong');
    assert.equal(receivedSources.diagnostics.response_mode, 'code_snippet');
    assert.equal(receivedSources.diagnostics.memory.history_injected, false);
    assert.equal(doneCalled, true);
    assert.match(calledUrl, /\/api\/v1\/query\/stream/);
    assert.equal(calledOptions.method, 'POST');
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('createProviderCredential skips encryption in local mode', async () => {
  const originalFetch = globalThis.fetch;
  let calledOptions = null;

  globalThis.fetch = async (url, options) => {
    calledOptions = options;
    return {
      ok: true,
      json: async () => ({ provider_credential: { id: 'local-1' } })
    };
  };

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => null,
    removeItem: () => null,
  };

  try {
    const { createProviderCredential } = await import('./api.js');
    await createProviderCredential({
      mode: 'local',
      provider: 'ollama',
      label: 'Local Dev',
      apiKey: 'should-not-be-encrypted-or-sent',
      model: 'qwen2.5-coder:3b',
      isActive: true,
    });
    const body = JSON.parse(calledOptions.body);
    assert.equal(body.mode, 'local');
    assert.equal(body.provider, 'ollama');
    assert.equal(body.encrypted_secret, undefined);
  } finally {
    globalThis.fetch = originalFetch;
    delete globalThis.localStorage;
  }
});

test('createProviderCredential encrypts secret in api mode', async () => {
  const originalFetch = globalThis.fetch;
  let calledUrl = null;
  let fetchCount = 0;
  let finalBody = null;

  globalThis.fetch = async (url, options) => {
    fetchCount++;
    // First call is to fetch public key for encryption
    if (url.includes('/api/v1/crypto/submission-key')) {
      return {
        ok: true,
        json: async () => ({
          key_id: 'test-key',
          algorithm: 'RSA-OAEP',
          public_key_pem: '-----BEGIN PUBLIC KEY-----\nMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC3zP8iN4ZqD3TqM/8k9nS5vO/W\n-----END PUBLIC KEY-----'
        })
      };
    }

    finalBody = JSON.parse(options.body);
    return {
      ok: true,
      json: async () => ({ provider_credential: { id: 'api-1' } })
    };
  };

  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => null,
    removeItem: () => null,
  };

  const originalCrypto = globalThis.crypto;
  Object.defineProperty(globalThis, 'crypto', {
    value: {
      subtle: {
        importKey: async () => ({}),
        encrypt: async () => new Uint8Array([1, 2, 3]).buffer,
      }
    },
    writable: true,
    configurable: true
  });

  const originalWindow = globalThis.window;
  globalThis.window = globalThis;

  try {
    const { createProviderCredential } = await import('./api.js');
    await createProviderCredential({
      mode: 'api',
      provider: 'aicredits',
      label: 'API Dev',
      apiKey: 'secret-api-key',
      model: 'gpt-4o',
      isActive: true,
    });
    assert.equal(finalBody.mode, 'api');
    assert.equal(finalBody.provider, 'aicredits');
    assert.ok(finalBody.encrypted_secret);
    assert.equal(finalBody.encrypted_secret.key_id, 'test-key');
  } finally {
    globalThis.fetch = originalFetch;
    Object.defineProperty(globalThis, 'crypto', {
      value: originalCrypto,
      writable: true,
      configurable: true
    });
    delete globalThis.localStorage;
    if (originalWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = originalWindow;
    }
  }
}
);

test('testEmbeddingConfig sends correct local payload without encryption', async (t) => {
  const originalFetch = globalThis.fetch;
  let fetchedUrl, fetchedOptions;

  globalThis.fetch = async (url, options) => {
    fetchedUrl = url;
    fetchedOptions = options;
    return {
      ok: true,
      json: async () => ({ ok: true, model: 'nomic-embed-text:latest', dimensions: 768 })
    };
  };

  try {
    const { testEmbeddingConfig } = await import('./api.js');
    const result = await testEmbeddingConfig({
      mode: 'local',
      provider: 'local',
      baseUrl: 'http://localhost:11434',
      model: 'nomic-embed-text:latest',
      dimensions: 768,
    });

    assert.equal(result.ok, true);
    assert.equal(result.model, 'nomic-embed-text:latest');
    assert.equal(result.dimensions, 768);

    assert.ok(fetchedUrl.endsWith('/api/v1/embedding/test'));
    const body = JSON.parse(fetchedOptions.body);
    assert.equal(body.mode, 'local');
    assert.equal(body.provider, 'local');
    assert.equal(body.base_url, 'http://localhost:11434');
    assert.equal(body.model, 'nomic-embed-text:latest');
    assert.equal(body.dimensions, 768);
    assert.equal(body.encrypted_secret, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('createProviderCredential in API mode with empty apiKey does not include encrypted_secret', async () => {
  const originalFetch = globalThis.fetch;
  let finalBody = null;
  globalThis.fetch = async (url, options) => {
    finalBody = JSON.parse(options.body);
    return {
      ok: true,
      json: async () => ({ provider_credential: { id: 'api-2' } })
    };
  };

  try {
    const { createProviderCredential } = await import('./api.js');
    await createProviderCredential({
      mode: 'api',
      provider: 'aicredits',
      label: 'API Dev',
      apiKey: '',
      model: 'gpt-4o',
      isActive: true,
    });
    assert.equal(finalBody.mode, 'api');
    assert.equal(finalBody.provider, 'aicredits');
    assert.equal(finalBody.encrypted_secret, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('saveEmbeddingConfig in API mode with empty apiKey does not include encrypted_secret', async () => {
  const originalFetch = globalThis.fetch;
  let finalBody = null;
  globalThis.fetch = async (url, options) => {
    finalBody = JSON.parse(options.body);
    return {
      ok: true,
      json: async () => ({ ok: true })
    };
  };

  try {
    const { saveEmbeddingConfig } = await import('./api.js');
    await saveEmbeddingConfig({
      mode: 'api',
      provider: 'openai_compatible',
      baseUrl: 'http://example',
      model: 'test',
      apiKey: '',
      dimensions: 256,
    });
    assert.equal(finalBody.mode, 'api');
    assert.equal(finalBody.provider, 'openai_compatible');
    assert.equal(finalBody.encrypted_secret, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
