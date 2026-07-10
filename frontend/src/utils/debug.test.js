import test from 'node:test';
import assert from 'node:assert/strict';

import { isDiagnosticsDebugEnabled, setDiagnosticsDebugEnabled } from './debug.js';

test('diagnostics debug flag is off by default', () => {
  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => null,
    removeItem: () => null,
  };

  try {
    assert.equal(isDiagnosticsDebugEnabled(), false);
  } finally {
    delete globalThis.localStorage;
  }
});

test('diagnostics debug flag persists to localStorage', () => {
  const storage = new Map();
  globalThis.localStorage = {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
  };

  try {
    setDiagnosticsDebugEnabled(true);
    assert.equal(isDiagnosticsDebugEnabled(), true);
    setDiagnosticsDebugEnabled(false);
    assert.equal(isDiagnosticsDebugEnabled(), false);
  } finally {
    delete globalThis.localStorage;
  }
});
