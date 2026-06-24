import { getBackendApiKey } from './storage.js';

const API_BASE = import.meta.env?.VITE_API_BASE_URL?.replace(/\/$/, "") || 'http://127.0.0.1:8000';

const authHeaders = (extra = {}) => {
  const headers = {
    'Content-Type': 'application/json',
    ...extra
  };
  const token = getBackendApiKey().trim();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const customKey = localStorage.getItem('CODESEEK_CUSTOM_ENCRYPTION_KEY');
  if (customKey) {
    headers['X-App-Encryption-Key'] = customKey.trim();
  }
  const modelOverride = localStorage.getItem('CODESEEK_ACTIVE_MODEL_OVERRIDE');
  if (modelOverride) {
    headers['X-App-Model-Override'] = modelOverride.trim();
  }
  return headers;
};

let submissionKeyPromise = null;

const readErrorDetail = async (res) => {
  try {
    const parsed = await res.json();
    return parsed.detail || parsed.message || '';
  } catch {
    return await res.text().catch(() => '');
  }
};

export const formatApiError = ({ action, status, detail = '' }) => {
  const normalizedDetail = `${detail}`.trim();
  if (action === 'Incremental indexing') {
    if (status === 403 || normalizedDetail.toLowerCase().includes('disabled')) {
      return 'Incremental indexing is not enabled on this server.';
    }
    if (status === 400 && normalizedDetail.toLowerCase().includes('plan unavailable')) {
      return 'Incremental preview is unavailable. Use Index latest instead.';
    }
    if (normalizedDetail.toLowerCase().includes('already in progress') || normalizedDetail.toLowerCase().includes('already running')) {
      return 'Indexing is already running.';
    }
    return 'Incremental indexing failed to start. Use Index latest as a fallback.';
  }
  if (status === 401) {
    if (normalizedDetail.toLowerCase().includes('authentication required')) {
      return `${action} failed (${status}): auth session expired. Sign in to GitHub again.`;
    }
    return `${action} failed (${status}): authentication failed. Check the backend API key and sign in again.`;
  }
  if (status === 429) {
    return `${action} failed (${status}): rate limit reached. Wait and retry, or switch provider credentials.`;
  }
  if (normalizedDetail.includes('No active provider credential configured')) {
    return `${action} failed (${status}): no active provider credential. Open API Config and add or activate a provider key.`;
  }
  if (normalizedDetail.includes('Session is not ready (status=failed)')) {
    return `${action} failed (${status}): indexing failed for this repo session. Retry indexing after checking GitHub access and backend logs.`;
  }
  if (normalizedDetail.includes('Session is not ready')) {
    return `${action} failed (${status}): repository indexing is still running. Wait for the session to become ready, then retry.`;
  }
  if (normalizedDetail.includes('Plaintext secret submission is disabled')) {
    return `${action} failed (${status}): secure secret submission requires a refresh. Reload the page and retry.`;
  }
  if (normalizedDetail.includes('GitHub token validation failed')) {
    return `${action} failed (${status}): GitHub rejected the token. Confirm repo scope and retry.`;
  }
  if (normalizedDetail.includes('GitHub OAuth')) {
    return `${action} failed (${status}): GitHub OAuth is misconfigured or unavailable. Check backend OAuth env vars and callback URL.`;
  }
  if (normalizedDetail.includes('GitHub repo fetch failed')) {
    return `${action} failed (${status}): GitHub repository listing failed. Reconnect GitHub and verify token scope.`;
  }
  if (normalizedDetail.includes('Provider API key rejected or lacks permission')) {
    return `${action} failed (${status}): provider rejected the configured API key or model access. Update the provider configuration and retry.`;
  }
  if (normalizedDetail.includes('Unsupported LLM provider configuration')) {
    return `${action} failed (${status}): provider configuration is invalid. Re-save the credential with a supported provider and model.`;
  }
  if (normalizedDetail.includes('Provider request timed out')) {
    return `${action} failed (${status}): provider request timed out. Retry or switch to a faster model.`;
  }
  if (normalizedDetail.includes('Provider request failed upstream')) {
    return `${action} failed (${status}): provider request failed upstream. Retry shortly or switch provider credentials.`;
  }
  return `${action} failed (${status})${normalizedDetail ? `: ${normalizedDetail}` : ''}`;
};

const throwApiError = async (action, res) => {
  const detail = await readErrorDetail(res);
  throw new Error(formatApiError({ action, status: res.status, detail }));
};

const pemToArrayBuffer = (pem) => {
  const base64 = pem
    .replace('-----BEGIN PUBLIC KEY-----', '')
    .replace('-----END PUBLIC KEY-----', '')
    .replace(/\s+/g, '');
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
};

const fetchSubmissionPublicKey = async () => {
  if (!submissionKeyPromise) {
    submissionKeyPromise = withNetworkError(
      async () => {
        const res = await fetch(`${API_BASE}/api/v1/crypto/submission-key`, {
          credentials: 'include',
        });
        if (!res.ok) {
          throw new Error(`Submission key fetch failed (${res.status})`);
        }
        return res.json();
      },
      'Submission key fetch'
    ).catch((err) => {
      submissionKeyPromise = null;
      throw err;
    });
  }
  return submissionKeyPromise;
};

const encryptSecretForSubmission = async (secret) => {
  const value = `${secret || ''}`.trim();
  if (!value) {
    throw new Error('Secret value cannot be empty.');
  }
  if (!window.crypto?.subtle) {
    throw new Error('Browser crypto support is unavailable for secure submission.');
  }
  const keyPayload = await fetchSubmissionPublicKey();
  const importedKey = await window.crypto.subtle.importKey(
    'spki',
    pemToArrayBuffer(keyPayload.public_key_pem),
    { name: 'RSA-OAEP', hash: 'SHA-256' },
    false,
    ['encrypt']
  );
  const ciphertext = await window.crypto.subtle.encrypt(
    { name: 'RSA-OAEP' },
    importedKey,
    new TextEncoder().encode(value)
  );
  const bytes = new Uint8Array(ciphertext);
  let binary = '';
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return {
    key_id: keyPayload.key_id,
    ciphertext: btoa(binary),
  };
};

const sendQuery = async (body) => {
  const res = await fetch(`${API_BASE}/api/v1/query`, {
    method: 'POST',
    credentials: 'include', headers: authHeaders(),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    await throwApiError('Query', res);
  }

  return res.json();
};

const withNetworkError = async (fn, label) => {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error(`${label} failed: backend unreachable at ${API_BASE}`);
    }
    throw err;
  }
};

/**
 * POST /api/v1/query
 * Sends a question for a specific repo and returns the answer + sources.
 */
export const queryRepo = async ({ question, repo_id }) => {
  return sendQuery({ question, repo_id, tenant_id: 'default' });
};

export const querySession = async ({ question, session_id, thread_id = '' }) => {
  return sendQuery({ question, session_id, thread_id: thread_id || undefined });
};

export const querySessionStream = async ({
  question,
  session_id,
  thread_id = '',
  onStatus,
  onDelta,
  onSources,
  onDone,
  onError,
  signal,
}) => {
  try {
    const res = await fetch(`${API_BASE}/api/v1/query/stream`, {
      method: 'POST',
      credentials: 'include',
      credentials: 'include', headers: authHeaders(),
      body: JSON.stringify({
        question,
        session_id,
        thread_id: thread_id || undefined,
      }),
      signal,
    });

    if (!res.ok) {
      const detail = await readErrorDetail(res);
      const errText = formatApiError({ action: 'Query stream', status: res.status, detail });
      throw new Error(errText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const event = JSON.parse(trimmed);
          if (event.type === 'status' && onStatus) {
            onStatus(event.message);
          } else if (event.type === 'delta' && onDelta) {
            onDelta(event.text);
          } else if (event.type === 'sources' && onSources) {
            onSources({
              sources: event.sources,
              context_tokens: event.context_tokens,
              diagnostics: event.diagnostics || null,
              evidence_confidence: event.evidence_confidence,
            });
          } else if (event.type === 'error' && onError) {
            onError(event.message);
          } else if (event.type === 'done' && onDone) {
            onDone(event);
          }
        } catch (e) {
          console.error('Failed to parse NDJSON line:', trimmed, e);
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      return;
    }
    if (onError) {
      onError(err.message || 'Stream connection closed unexpectedly');
    }
  }
};

export const createSession = async ({ repoFullName, repoUrl, tenantId = 'local', githubToken = '', enableChunkDescriptions = false }) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
        body: JSON.stringify({
          repo_full_name: repoFullName,
          repo_url: repoUrl,
          tenant_id: tenantId,
          github_token: githubToken,
          enable_chunk_descriptions: enableChunkDescriptions,
        }),
      }),
    'Session create'
  );

  if (!res.ok) {
    await throwApiError('Session create', res);
  }

  const data = await res.json();
  return data.session;
};

export const listSessions = async () => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'List sessions'
  );
  if (!res.ok) await throwApiError('List sessions', res);
  const data = await res.json();
  return data.sessions || [];
};

export const deleteSessionApi = async (sessionId) => {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`, {
    method: 'DELETE',
    credentials: 'include', headers: authHeaders(),
  });
  if (!res.ok) await throwApiError('Delete session', res);
  return res.json();
};

export const retrySessionIndexing = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/retry`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Retry indexing'
  );
  if (!res.ok) await throwApiError('Retry indexing', res);
  const data = await res.json();
  return data.session;
};


export const fetchSessionMessages = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/messages`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch session messages'
  );
  if (!res.ok) await throwApiError('Fetch session messages', res);
  const data = await res.json();
  return data.messages || [];
};

export const listSessionThreads = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/threads`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'List session threads'
  );
  if (!res.ok) await throwApiError('List session threads', res);
  const data = await res.json();
  return data.threads || [];
};

export const createSessionThread = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/threads`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Create session thread'
  );
  if (!res.ok) await throwApiError('Create session thread', res);
  const data = await res.json();
  return data.thread;
};

export const fetchThreadMessages = async (threadId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/threads/${threadId}/messages`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch thread messages'
  );
  if (!res.ok) await throwApiError('Fetch thread messages', res);
  const data = await res.json();
  return data.messages || [];
};

export const clearThreadMessagesApi = async (threadId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/threads/${threadId}/messages`, {
        method: 'DELETE',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Clear thread messages'
  );
  if (!res.ok) await throwApiError('Clear thread messages', res);
  return res.json();
};

export const clearSessionMessagesApi = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/messages`, {
        method: 'DELETE',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Clear session messages'
  );
  if (!res.ok) await throwApiError('Clear session messages', res);
  return res.json();
};

/**
 * GET /api/v1/health
 * Returns true if backend is alive, false otherwise.
 */
export const fetchHealth = async () => {
  try {
    const res = await fetch(`${API_BASE}/api/v1/health`, {
      credentials: 'include',
      headers: authHeaders(),
    });
    return res.ok;
  } catch {
    return false;
  }
};

/**
 * POST /auth/github
 * Exchange GitHub OAuth code via the backend and create a server-side session.
 */
export const exchangeGithubCode = async (code) => {
  const res = await fetch(`${API_BASE}/auth/github`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });

  if (!res.ok) {
    await throwApiError('GitHub auth', res);
  }

  return res.json();
};

export const connectGithubToken = async (accessToken) => {
  const encryptedSecret = await encryptSecretForSubmission(accessToken);
  const res = await fetch(`${API_BASE}/auth/github/token`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ encrypted_secret: encryptedSecret }),
  });

  if (!res.ok) {
    await throwApiError('GitHub token connect', res);
  }

  return res.json();
};

export const listGithubRepos = async () => {
  const res = await fetch(`${API_BASE}/api/v1/github/repos`, {
    credentials: 'include',
  });
  if (!res.ok) {
    await throwApiError('GitHub repo list', res);
  }
  const data = await res.json();
  return data.repos || [];
};

export const listProviderCredentials = async () => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/provider-credentials`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'List provider credentials'
  );
  if (!res.ok) {
    await throwApiError('List provider credentials', res);
  }
  const data = await res.json();
  const list = data.provider_credentials || [];
  return list.map((c) => ({
    ...c,
    isActive: !!c.is_active,
  }));
};

export const createProviderCredential = async ({ provider, label, apiKey, model = '', isActive }) => {
  const normalizedProvider = `${provider || ''}`.trim().toLowerCase();
  const normalizedApiKey = `${apiKey || ''}`.trim();
  const encryptedSecret =
    normalizedProvider === 'local' && !normalizedApiKey
      ? null
      : await encryptSecretForSubmission(normalizedApiKey);
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/provider-credentials`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
        body: JSON.stringify({
          provider: normalizedProvider || provider,
          label,
          encrypted_secret: encryptedSecret || undefined,
          model,
          is_active: isActive,
        }),
      }),
    'Create provider credential'
  );
  if (!res.ok) {
    await throwApiError('Create provider credential', res);
  }
  const data = await res.json();
  const cred = data.provider_credential;
  return {
    ...cred,
    isActive: !!cred.is_active,
  };
};

export const activateProviderCredential = async (credentialId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/provider-credentials/${credentialId}/activate`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Activate provider credential'
  );
  if (!res.ok) {
    await throwApiError('Activate provider credential', res);
  }
  const data = await res.json();
  const cred = data.provider_credential;
  return {
    ...cred,
    isActive: !!cred.is_active,
  };
};

export const deleteProviderCredential = async (credentialId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/provider-credentials/${credentialId}`, {
        method: 'DELETE',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Delete provider credential'
  );
  if (!res.ok) {
    await throwApiError('Delete provider credential', res);
  }
  return res.json();
};

export const fetchGithubSessionMe = async () => {
  const res = await fetch(`${API_BASE}/auth/me`, {
    credentials: 'include',
  });
  if (!res.ok) {
    await throwApiError('Auth me', res);
  }
  return res.json();
};

export const logoutGithubSession = async () => {
  const res = await fetch(`${API_BASE}/auth/logout`, {
    method: 'POST',
    credentials: 'include',
  });
  if (!res.ok) {
    await throwApiError('Auth logout', res);
  }
  return res.json();
};

export const fetchSessionRepoStatus = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/repo-status`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch session repo status'
  );
  if (!res.ok) await throwApiError('Fetch session repo status', res);
  return res.json();
};

export const fetchSessionFreshness = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/freshness`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch session freshness status'
  );
  if (!res.ok) await throwApiError('Fetch session freshness status', res);
  return res.json();
};

export const indexLatestVersion = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/index-latest`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Index latest version'
  );
  if (!res.ok) await throwApiError('Index latest version', res);
  return res.json();
};

export const indexLatestSession = indexLatestVersion;

export const fetchLatestEvaluationReport = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/evaluation/latest`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch latest evaluation report'
  );
  if (!res.ok) await throwApiError('Fetch latest evaluation report', res);
  return res.json();
};

export const fetchEvaluationRegressionTests = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/evaluation/regression-tests`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch evaluation regression tests'
  );
  if (!res.ok) await throwApiError('Fetch evaluation regression tests', res);
  return res.json();
};

export const fetchLatestGlobalEvaluationReport = async () => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/evals/latest`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch latest global evaluation report'
  );
  if (!res.ok) await throwApiError('Fetch latest global evaluation report', res);
  return res.json();
};


export const fetchIndexPreview = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/index-preview`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch index preview'
  );
  if (!res.ok) await throwApiError('Fetch index preview', res);
  return res.json();
};


export const indexSessionIncremental = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/index-incremental`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Incremental indexing'
  );
  if (!res.ok) await throwApiError('Incremental indexing', res);
  return res.json();
};

export const fetchLatestIndexingJob = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/indexing-job/latest`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Fetch latest indexing job'
  );
  if (!res.ok) await throwApiError('Fetch latest indexing job', res);
  return res.json();
};

export const cancelLatestIndexingJob = async (sessionId) => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/sessions/${sessionId}/indexing-job/cancel`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Cancel indexing job'
  );
  if (!res.ok) await throwApiError('Cancel indexing job', res);
  return res.json();
};

export const fetchIndexingJobHistory = async (sessionId, limit = 20) => {
  const params = new URLSearchParams({ limit: String(limit) });
  const res = await withNetworkError(
    () =>
      fetch(
        `${API_BASE}/api/v1/sessions/${sessionId}/indexing-jobs?${params}`,
        {
          credentials: 'include',
          credentials: 'include', headers: authHeaders(),
        }
      ),
    'Fetch indexing job history'
  );
  return res.json();
};

export const getEmbeddingConfig = async () => {
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/embedding/config`, {
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
      }),
    'Get embedding config'
  );
  if (!res.ok) await throwApiError('Get embedding config', res);
  return res.json();
};

export const saveEmbeddingConfig = async (payload) => {
  const { provider, baseUrl, model, apiKey, dimensions, timeoutSeconds, batchSize } = payload;
  let encryptedSecret = null;
  if (apiKey) {
    encryptedSecret = await encryptSecretForSubmission(apiKey);
  }
      
  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/embedding/config`, {
        method: 'PUT',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
        body: JSON.stringify({
          provider,
          base_url: baseUrl,
          model,
          encrypted_secret: encryptedSecret || undefined,
          dimensions,
          timeout_seconds: timeoutSeconds,
          batch_size: batchSize,
        }),
      }),
    'Save embedding config'
  );
  if (!res.ok) await throwApiError('Save embedding config', res);
  return res.json();
};

export const testEmbeddingConfig = async (payload) => {
  const { provider, baseUrl, model, apiKey, dimensions } = payload;
  let encryptedSecret = null;
  if (apiKey) {
    encryptedSecret = await encryptSecretForSubmission(apiKey);
  }

  const res = await withNetworkError(
    () =>
      fetch(`${API_BASE}/api/v1/embedding/test`, {
        method: 'POST',
        credentials: 'include',
        credentials: 'include', headers: authHeaders(),
        body: JSON.stringify({
          provider,
          base_url: baseUrl,
          model,
          encrypted_secret: encryptedSecret || undefined,
          dimensions,
        }),
      }),
    'Test embedding config'
  );
  if (!res.ok) await throwApiError('Test embedding config', res);
  return res.json();
};

export const getEmbeddingOptions = async () => {
  const res = await withNetworkError(
    () => fetch(`${API_BASE}/api/v1/embedding/options`, { credentials: 'include', headers: authHeaders() }),
    'Get embedding options'
  );
  if (!res.ok) await throwApiError('Get embedding options', res);
  return res.json();
};
