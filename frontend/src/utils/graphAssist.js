const GLOBAL_KEY = 'codeseek.graphAssist.enabled';

function sessionKey(sessionId) {
  const id = String(sessionId || '').trim();
  return id ? `codeseek.graphAssist.${id}` : GLOBAL_KEY;
}

function normalizeStoredFlag(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase());
}

export function getGraphAssistEnabled(sessionId = '') {
  if (typeof localStorage === 'undefined') return false;
  try {
    const scoped = localStorage.getItem(sessionKey(sessionId));
    if (scoped !== null) return normalizeStoredFlag(scoped);
    return normalizeStoredFlag(localStorage.getItem(GLOBAL_KEY));
  } catch {
    return false;
  }
}

export function setGraphAssistEnabled(sessionId = '', enabled = false) {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(sessionKey(sessionId), enabled ? '1' : '0');
  } catch {
    // Ignore storage failures; the in-memory React state still controls this session.
  }
}

export function graphRetrievalModeForEnabled(enabled) {
  return enabled ? 'graph_assist' : 'standard';
}
