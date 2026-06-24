/**
 * Shared test helpers and fixtures for Codeseek E2E specs.
 */

import * as fs from 'fs';
import * as path from 'path';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.CODESEEK_API_KEY || '';
const GITHUB_PAT = process.env.GITHUB_TEST_PAT || '';
export const TEST_REPO = process.env.TEST_REPO || 'octocat/Hello-World';

/** GET /auth/me — returns parsed body or null */
export async function getAuthMe() {
  const res = await fetch(`${BACKEND_URL}/auth/me`, { credentials: 'include' });
  if (!res.ok) return null;
  return res.json();
}

/** Backend health check */
export async function backendIsHealthy() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/health`, {
      headers: { Authorization: `Bearer ${API_KEY}` },
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Connect GitHub via PAT (encrypted submission) directly via the backend API.
 * Used to seed auth state before tests that need a logged-in user.
 */
export async function connectGithubViaPat(page) {
  if (!GITHUB_PAT) throw new Error('GITHUB_TEST_PAT env var is required for auth tests');

  // 1. Fetch the RSA submission key
  const keyRes = await fetch(`${BACKEND_URL}/api/v1/crypto/submission-key`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
  if (!keyRes.ok) throw new Error(`Submission key fetch failed: ${keyRes.status}`);
  const { public_key_pem, key_id } = await keyRes.json();

  // 2. Encrypt the PAT using the browser's SubtleCrypto (via page.evaluate)
  const encrypted = await page.evaluate(
    async ({ pem, pat }) => {
      const base64 = pem
        .replace('-----BEGIN PUBLIC KEY-----', '')
        .replace('-----END PUBLIC KEY-----', '')
        .replace(/\s+/g, '');
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      const importedKey = await crypto.subtle.importKey(
        'spki',
        bytes.buffer,
        { name: 'RSA-OAEP', hash: 'SHA-256' },
        false,
        ['encrypt']
      );
      const ciphertext = await crypto.subtle.encrypt(
        { name: 'RSA-OAEP' },
        importedKey,
        new TextEncoder().encode(pat)
      );
      const outBytes = new Uint8Array(ciphertext);
      let bin = '';
      outBytes.forEach((b) => (bin += String.fromCharCode(b)));
      return btoa(bin);
    },
    { pem: public_key_pem, pat: GITHUB_PAT }
  );

  // 3. POST encrypted PAT to backend — backend sets auth session cookie
  const tokenRes = await page.request.post(`${BACKEND_URL}/auth/github/token`, {
    data: { encrypted_secret: { key_id, ciphertext: encrypted } },
    headers: { 'Content-Type': 'application/json' },
  });
  if (!tokenRes.ok) {
    const body = await tokenRes.json().catch(() => ({}));
    throw new Error(`GitHub PAT connect failed: ${tokenRes.status} — ${body.detail || ''}`);
  }
  return tokenRes.json();
}

/**
 * Create a provider credential directly via API.
 * Returns the created credential object.
 */
export async function createProviderCredentialViaApi(page, { provider, label, apiKey, model = '' }) {
  // Encrypt the API key
  const keyRes = await fetch(`${BACKEND_URL}/api/v1/crypto/submission-key`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
  const { public_key_pem, key_id } = await keyRes.json();

  const encrypted = await page.evaluate(
    async ({ pem, secret }) => {
      const base64 = pem
        .replace('-----BEGIN PUBLIC KEY-----', '')
        .replace('-----END PUBLIC KEY-----', '')
        .replace(/\s+/g, '');
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const key = await crypto.subtle.importKey(
        'spki', bytes.buffer, { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['encrypt']
      );
      const ct = await crypto.subtle.encrypt({ name: 'RSA-OAEP' }, key, new TextEncoder().encode(secret));
      const out = new Uint8Array(ct);
      let bin = '';
      out.forEach((b) => (bin += String.fromCharCode(b)));
      return btoa(bin);
    },
    { pem: public_key_pem, secret: apiKey }
  );

  const res = await page.request.post(`${BACKEND_URL}/api/v1/provider-credentials`, {
    data: {
      provider,
      label,
      encrypted_secret: { key_id, ciphertext: encrypted },
      model,
      is_active: true,
    },
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(`Provider credential create failed: ${res.status} — ${body.detail || ''}`);
  }
  const data = await res.json();
  return data.provider_credential;
}

/**
 * List sessions via API.
 */
export async function listSessionsViaApi() {
  const res = await fetch(`${BACKEND_URL}/api/v1/sessions`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`List sessions failed: ${res.status}`);
  const data = await res.json();
  return data.sessions || [];
}

/**
 * Delete all sessions via API (test teardown helper).
 */
export async function deleteAllSessionsViaApi() {
  const sessions = await listSessionsViaApi();
  for (const session of sessions) {
    await fetch(`${BACKEND_URL}/api/v1/sessions/${session.id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${API_KEY}` },
      credentials: 'include',
    }).catch(() => {});
  }
}

/**
 * Wait for a session to reach a terminal status (ready or failed).
 * Polls every 3s, times out after maxWaitMs.
 */
export async function waitForSessionReady(sessionId, maxWaitMs = 300_000) {
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    const sessions = await listSessionsViaApi();
    const session = sessions.find((s) => s.id === sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === 'ready') return session;
    if (session.status === 'failed') throw new Error(`Session ${sessionId} failed: ${session.error}`);
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`Session ${sessionId} did not reach 'ready' within ${maxWaitMs}ms`);
}
