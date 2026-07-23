import React from 'react';
import { useState, useEffect, useRef } from 'react';
import {
  createProviderCredential,
  listProviderCredentials,
  saveEmbeddingConfig,
  getEmbeddingConfig,
  testEmbeddingConfig,
  deleteProviderCredential,
} from '../utils/api';
import { defaultEmbeddingModelForMode } from '../utils/validation.js';

export default function ApiTokensModal({ onClose }) {
  const [mode, setMode] = useState('api');
  const [profiles, setProfiles] = useState({
    api: {
      providerUrl: '',
      apiKey: '',
      llmModel: '',
      embModel: defaultEmbeddingModelForMode('api'),
      hasProviderSecret: false,
      hasEmbeddingSecret: false,
    },
    local: {
      providerUrl: 'http://localhost:11434',
      apiKey: '',
      llmModel: 'qwen2.5-coder:3b',
      embModel: defaultEmbeddingModelForMode('local'),
      hasProviderSecret: false,
      hasEmbeddingSecret: false,
    }
  });

  const currentProfile = profiles[mode];

  const handleProfileChange = (field, value) => {
    setProfiles(prev => ({
      ...prev,
      [mode]: {
        ...prev[mode],
        [field]: value
      }
    }));
  };

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
          const apiLlm = llms.find(t => t.provider !== 'local') || llms[0];
          const localLlm = llms.find(t => t.provider === 'local');
          const activeLlm = llms.find(t => t.isActive) || llms[0];

          if (activeLlm) {
            setActiveConfig(activeLlm);
          }
          if (emb) {
            setActiveEmbConfig(emb);
            setMode('api');
          }

          setProfiles(prev => {
            const next = { ...prev };

            // Populate API profile
            if (apiLlm) {
              next.api.llmModel = apiLlm.model || '';
              next.api.hasProviderSecret = apiLlm.has_secret || false;
            }
            if (emb && emb.profiles && emb.profiles.api) {
              const apiEmb = emb.profiles.api;
              next.api.providerUrl = apiEmb.base_url || '';
              next.api.embModel = defaultEmbeddingModelForMode('api');
              next.api.hasEmbeddingSecret = apiEmb.has_secret || false;
            } else if (emb && emb.mode !== 'local') {
              next.api.providerUrl = emb.base_url || '';
              next.api.embModel = defaultEmbeddingModelForMode('api');
              next.api.hasEmbeddingSecret = emb.has_secret || false;
            }

            // Populate Local profile
            if (localLlm) {
              next.local.llmModel = localLlm.model || 'qwen2.5-coder:3b';
              next.local.hasProviderSecret = localLlm.has_secret || false;
            }
            if (emb && emb.profiles && emb.profiles.local) {
              const localEmb = emb.profiles.local;
              next.local.providerUrl = localEmb.base_url || 'http://localhost:11434';
              next.local.embModel = defaultEmbeddingModelForMode('local');
              next.local.hasEmbeddingSecret = localEmb.has_secret || false;
            } else if (emb && emb.mode === 'local') {
              next.local.providerUrl = emb.base_url || 'http://localhost:11434';
              next.local.embModel = defaultEmbeddingModelForMode('local');
              next.local.hasEmbeddingSecret = emb.has_secret || false;
            }

            return next;
          });
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
    let url = currentProfile.providerUrl.trim();

    if (!url) {
      setError('Provider URL is required.');
      return false;
    }

    if (!url.startsWith('https://') && !url.startsWith('http://')) {
      setError('Provider URL must start with https:// or http://');
      return false;
    }

    if (mode === 'api') {
      const hasUsableCloudKey =
        currentProfile.apiKey.trim() ||
        currentProfile.hasProviderSecret ||
        currentProfile.hasEmbeddingSecret;

      if (!hasUsableCloudKey) {
        setError('API Key is required for API Provider mode.');
        return false;
      }
    }
    if (!currentProfile.llmModel.trim()) {
      setError('LLM Model is required.');
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
        mode,
        provider: mode === 'local' ? 'local' : 'openai_compatible',
        baseUrl: currentProfile.providerUrl.trim().replace(/\/+$/, ''),
        model: defaultEmbeddingModelForMode(mode),
        apiKey: currentProfile.apiKey.trim(),
      };
      const result = await testEmbeddingConfig(payload);
      testMsg = `Embedding config tested successfully with ${result.model || defaultEmbeddingModelForMode(mode)}. LLM test skipped.`;
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
    let url = currentProfile.providerUrl.trim().replace(/\/+$/, '');

    try {
      // Save LLM credential
      const created = await createProviderCredential({
        mode,
        provider: mode === 'local' ? 'local' : 'aicredits',
        label: mode === 'local' ? 'Local Provider' : 'OpenAI-Compatible Provider',
        apiKey: currentProfile.apiKey.trim(),
        model: currentProfile.llmModel.trim(),
        isActive: true,
      });
      setActiveConfig(created);

      handleProfileChange('hasProviderSecret', true);
      handleProfileChange('apiKey', '');

      // Save Embedding config
      const payload = {
        mode,
        provider: mode === 'local' ? 'local' : 'openai_compatible',
        baseUrl: url,
        model: defaultEmbeddingModelForMode(mode),
        apiKey: currentProfile.apiKey.trim(),
      };
      const updatedEmb = await saveEmbeddingConfig(payload);
      setActiveEmbConfig(updatedEmb);

      handleProfileChange('hasEmbeddingSecret', true);

      setSuccessMsg('Configuration saved successfully. The default embedding model is used for indexing.');
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
                  <div className="flex">
                    <span className="w-24 text-text-muted">Embedding:</span>
                    <span className="font-mono">{defaultEmbeddingModelForMode(mode)}</span>
                  </div>
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
                value={currentProfile.providerUrl}
                onChange={(e) => handleProfileChange('providerUrl', e.target.value)}
                placeholder="https://api.openai.com/v1"
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                required
              />
              {mode === 'api' && <p className="text-[11px] text-text-muted mt-0.5">Must be OpenAI-compatible and include https://</p>}
            </div>

            {mode === 'api' && (
              <div className="flex flex-col gap-1">
                <label className="text-2xs font-mono text-text-muted uppercase">API Key</label>
                <input
                  type="password"
                  value={currentProfile.apiKey}
                  onChange={(e) => handleProfileChange('apiKey', e.target.value)}
                  placeholder={(currentProfile.hasProviderSecret || currentProfile.hasEmbeddingSecret) ? "•••••••• (Saved)" : "Enter API key"}
                  className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                  required={!(currentProfile.hasProviderSecret || currentProfile.hasEmbeddingSecret)}
                />
                <p className="text-[11px] text-text-muted mt-0.5">Used securely for LLM and embedding requests.</p>
              </div>
            )}

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">LLM Model</label>
              <input
                type="text"
                value={currentProfile.llmModel}
                onChange={(e) => handleProfileChange('llmModel', e.target.value)}
                placeholder="e.g. gpt-4o-mini or deepseek/deepseek-v4-flash"
                className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-primary placeholder-text-muted font-mono focus:outline-none focus:border-text-muted"
                required
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-2xs font-mono text-text-muted uppercase">Embedding Model</label>
              <div className="bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs text-text-secondary font-mono">
                {defaultEmbeddingModelForMode(mode)}
              </div>
              <p className="text-[11px] text-text-muted mt-0.5">CodeSeek uses this default embedding model for repository indexing.</p>
            </div>
          </form>
        </div>

        {/* Error/Success notification banner */}
        {(error || successMsg) && (
          <div className={`border-t border-b px-4 py-3 flex items-start gap-3 animate-fadeIn ${error ? 'bg-offline/10 border-offline/20' : 'bg-online/10 border-online/20'
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
            className="px-4 py-2 bg-surface-2 hover:bg-surface-1 border border-border text-text-primary text-xs font-mono rounded-lg transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
