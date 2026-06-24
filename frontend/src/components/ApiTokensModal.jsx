import { useState, useEffect, useRef } from 'react';
import {
  createProviderCredential,
  listProviderCredentials,
  getEmbeddingConfig,
  saveEmbeddingConfig,
  testEmbeddingConfig,
  deleteProviderCredential,
} from '../utils/api';

export default function ApiTokensModal({ onClose }) {
  const [providerUrl, setProviderUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [llmModel, setLlmModel] = useState('');
  const [embModel, setEmbModel] = useState('');
  const [embDims, setEmbDims] = useState('');

  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  const [activeConfig, setActiveConfig] = useState(null);
  const [activeEmbConfig, setActiveEmbConfig] = useState(null);

  const overlayRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const loadData = async () => {
      try {
        const [llms, emb] = await Promise.all([
          listProviderCredentials(),
          getEmbeddingConfig().catch(() => null),
        ]);
        if (!cancelled) {
          const activeLlm = llms.find(t => t.isActive) || llms[0];
          if (activeLlm) {
            setActiveConfig(activeLlm);
            setLlmModel(activeLlm.model || '');
          }
          if (emb) {
            setActiveEmbConfig(emb);
            setProviderUrl(emb.base_url || '');
            setEmbModel(emb.model || '');
            setEmbDims(emb.dimensions ? String(emb.dimensions) : '');
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load configurations.');
      }
    };
    loadData();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const validate = () => {
    setError(null);
    setSuccessMsg(null);
    let url = providerUrl.trim();
    if (!url.startsWith('https://') && !url.startsWith('http://')) {
      setError('Provider URL must start with https:// or http://');
      return false;
    }
    if (!apiKey.trim() && !activeEmbConfig?.api_key_configured && !activeConfig) {
      setError('API Key is required.');
      return false;
    }
    if (!llmModel.trim()) {
      setError('LLM Model is required.');
      return false;
    }
    if (!embModel.trim()) {
      setError('Embedding Model is required.');
      return false;
    }
    return true;
  };

  const handleTest = async () => {
    if (!validate()) return;
    setTesting(true);
    let testMsg = '';

    try {
      const payload = {
        provider: 'openai_compatible',
        baseUrl: providerUrl.trim().replace(/\/+$/, ''),
        model: embModel.trim(),
        apiKey: apiKey.trim(),
        dimensions: embDims ? parseInt(embDims, 10) : undefined,
      };
      const result = await testEmbeddingConfig(payload);
      testMsg = `Embedding config tested successfully! Dimensions: ${result.dimensions}, Model: ${result.model}. LLM test skipped.`;
      setSuccessMsg(testMsg);
    } catch (err) {
      setError(`Embedding test failed: ${err.message || 'Unknown error'}`);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!validate()) return;
    setSaving(true);
    let url = providerUrl.trim().replace(/\/+$/, '');

    try {
      // Save LLM credential
      const created = await createProviderCredential({
        provider: 'aicredits',
        label: 'OpenAI-Compatible Provider',
        apiKey: apiKey.trim(),
        model: llmModel.trim(),
        isActive: true,
      });
      setActiveConfig(created);

      // Save Embedding config
      const payload = {
        provider: 'openai_compatible',
        baseUrl: url,
        model: embModel.trim(),
        apiKey: apiKey.trim(),
        dimensions: embDims ? parseInt(embDims, 10) : undefined,
      };
      const updatedEmb = await saveEmbeddingConfig(payload);
      setActiveEmbConfig(updatedEmb);

      setSuccessMsg('Configuration saved successfully. Note: changing embedding settings may require reindexing active sessions.');
      window.dispatchEvent(new Event('CODESEEK_PROVIDER_CHANGED'));
    } catch (err) {
      if (err.message && err.message.includes('401')) {
        setError('Provider rejected the API key. Check the key and provider account access.');
      } else {
        setError(err.message || 'Failed to save configuration.');
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 bg-black/60 flex items-start justify-center pt-[10vh]"
      onClick={(e) => e.target === overlayRef.current && onClose()}
    >
      <div className="bg-surface-2 border border-border rounded-2xl w-full max-w-lg mx-4 shadow-xl animate-fadeIn flex flex-col max-h-[85vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-2 border-b border-border shrink-0 bg-surface-2">
          <div>
            <h2 className="text-sm font-medium text-text-primary flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-online" />
              Model Provider
            </h2>
            <p className="text-2xs text-text-muted mt-1 font-mono">Use one OpenAI-compatible provider for both answers and embeddings.</p>
          </div>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors text-lg leading-none self-start"
          >
            ×
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-4">
          <form id="provider-form" onSubmit={handleSave} className="space-y-4">

            {(activeConfig || activeEmbConfig) && (
              <div className="bg-surface-3 border border-border rounded-xl p-3 mb-4">
                <h3 className="text-2xs font-mono text-text-secondary uppercase tracking-wider mb-2">Active Config</h3>
                <div className="text-xs text-text-primary space-y-1">
                  {activeEmbConfig?.base_url && (
                    <div className="flex">
                      <span className="w-24 text-text-muted">Provider:</span>
                      <span className="font-mono truncate">{new URL(activeEmbConfig.base_url).host}</span>
                    </div>
                  )}
                  {activeConfig?.model && (
                    <div className="flex">
                      <span className="w-24 text-text-muted">LLM Model:</span>
                      <span className="font-mono">{activeConfig.model}</span>
                    </div>
                  )}
                  {activeEmbConfig?.model && (
                    <div className="flex">
                      <span className="w-24 text-text-muted">Emb Model:</span>
                      <span className="font-mono">{activeEmbConfig.model}</span>
                    </div>
                  )}
                  <div className="flex mt-2">
                    <span className="w-24 text-text-muted">Emb Valid:</span>
                    <span className={`font-mono ${activeEmbConfig?.model ? 'text-online' : 'text-warning'}`}>
                      {activeEmbConfig?.model ? 'Yes' : 'Unknown'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">Provider API URL</label>
              <input
                type="text"
                value={providerUrl}
                onChange={(e) => setProviderUrl(e.target.value)}
                placeholder="https://api.aicredits.in/v1"
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                required
              />
              <p className="text-[11px] text-text-muted mt-0.5">Must be OpenAI-compatible and include https://</p>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">API Key</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={(activeConfig || activeEmbConfig) ? "•••••••• (Saved)" : "Enter API key"}
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                required={!(activeConfig || activeEmbConfig)}
              />
              <p className="text-[11px] text-text-muted mt-0.5">Used securely for LLM and embedding requests.</p>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">LLM Model</label>
              <input
                type="text"
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                placeholder="e.g. gpt-4o-mini or deepseek/deepseek-v4-flash"
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                required
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">Embedding Model</label>
              <input
                type="text"
                value={embModel}
                onChange={(e) => setEmbModel(e.target.value)}
                placeholder="e.g. text-embedding-3-small"
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                required
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">Embedding Dimensions</label>
              <input
                type="number"
                value={embDims}
                onChange={(e) => setEmbDims(e.target.value)}
                placeholder="Optional (e.g. 1536)"
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
              />
              <p className="text-[11px] text-text-muted mt-0.5">Leave empty unless your embedding model requires a fixed dimension.</p>
            </div>
          </form>
        </div>

        {/* Error/Success notification banner */}
        {(error || successMsg) && (
          <div className={`border-t border-b px-4 py-3 flex items-start gap-3 animate-fadeIn ${
            error ? 'bg-offline/10 border-offline/20' : 'bg-online/10 border-online/20'
          }`}>
            <p className={`text-xs font-mono leading-relaxed flex-1 ${error ? 'text-offline/90' : 'text-online/90'}`}>
              {error ? `⚠ ${error}` : `✓ ${successMsg}`}
            </p>
            <button
              type="button"
              onClick={() => { setError(null); setSuccessMsg(null); }}
              className="text-text-muted hover:text-text-primary transition-colors text-lg font-bold leading-none shrink-0 -mt-1"
              title="Dismiss"
            >
              ×
            </button>
          </div>
        )}

        {/* Footer Actions */}
        <div className="p-4 border-t border-border bg-surface-3 flex justify-end gap-3 shrink-0">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-transparent hover:bg-surface-2 text-text-secondary text-xs font-mono rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleTest}
            disabled={testing || saving}
            className="px-4 py-2 bg-surface-2 hover:bg-surface-1 border border-border text-text-primary text-xs font-mono rounded-lg transition-colors disabled:opacity-50"
          >
            {testing ? 'Testing...' : 'Test'}
          </button>
          <button
            type="submit"
            form="provider-form"
            disabled={saving || testing}
            className="px-4 py-2 bg-text-primary hover:bg-text-secondary text-surface-0 text-xs font-semibold font-mono tracking-wider rounded-lg transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
