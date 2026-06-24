const STORAGE_KEY = 'CODESEEK_DEBUG_DIAGNOSTICS';

function normalizeFlag(value) {
  return `${value ?? ''}`.trim().toLowerCase();
}

export function isDiagnosticsDebugEnabled() {
  const envFlag = normalizeFlag(import.meta.env?.VITE_ENABLE_DEBUG_DIAGNOSTICS);
  if (envFlag && ['1', 'true', 'yes', 'on'].includes(envFlag)) {
    return true;
  }
  if (typeof localStorage === 'undefined') {
    return false;
  }
  return ['1', 'true', 'yes', 'on'].includes(normalizeFlag(localStorage.getItem(STORAGE_KEY)));
}

export function setDiagnosticsDebugEnabled(enabled) {
  if (typeof localStorage === 'undefined') {
    return;
  }
  if (enabled) {
    localStorage.setItem(STORAGE_KEY, '1');
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}
