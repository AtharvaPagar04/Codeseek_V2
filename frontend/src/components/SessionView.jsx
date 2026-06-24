import { useState, useEffect, useRef } from 'react';

import MessageBubble from './MessageBubble';
import EmptyState from './EmptyState';
import ConfirmDialog from './ConfirmDialog';
import IndexingLiveLog from './IndexingLiveLog';
import { useChat } from '../hooks/useChat';

import { listProviderCredentials, fetchSessionRepoStatus, fetchSessionFreshness, indexLatestVersion, fetchIndexPreview, indexSessionIncremental, fetchLatestIndexingJob, cancelLatestIndexingJob, fetchIndexingJobHistory } from '../utils/api';


function getProviderFallbackModel(provider) {
  if (provider === 'groq') return 'llama-3.3-70b-versatile';
  if (provider === 'openai') return 'gpt-4o-mini';
  if (provider === 'openrouter') return 'openai/gpt-4o-mini';
  if (provider === 'aicredits') return 'gpt-5.4-mini';
  if (provider === 'local') return 'auto';
  return 'gemini-2.0-flash';
}

export default function SessionView({
  session,
  appendMessage,
  onClearMessages,
  onRetryIndexing,
  onCancelSession,
  updateSession,
}) {
  const [input, setInput] = useState('');
  const [confirmClear, setConfirmClear] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [copiedSession, setCopiedSession] = useState(false);
  const [activeProvider, setActiveProvider] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [checkingStatus, setCheckingStatus] = useState(false);
  const [dismissedFreshnessPrompt, setDismissedFreshnessPrompt] = useState(false);
  const [showUpToDatePopup, setShowUpToDatePopup] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const [reindexingError, setReindexingError] = useState(null);
  const [latestJob, setLatestJob] = useState(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [cancelMessage, setCancelMessage] = useState(null);
  const [jobHistory, setJobHistory] = useState([]);
  const [showJobHistory, setShowJobHistory] = useState(false);
  const [loadingJobHistory, setLoadingJobHistory] = useState(false);
  const bottomRef = useRef(null);

  const repoStatus = session.repo_status;
  const freshness = session.freshness;
  const freshnessStatus = freshness?.freshness_status || (session.status === 'indexing' ? 'indexing' : (repoStatus?.status === 'up_to_date' ? 'latest' : (repoStatus?.status === 'out_of_date' ? 'stale_commit' : repoStatus?.status)));

  const fetchLatestJobData = async () => {
    try {
      const data = await fetchLatestIndexingJob(session.id);
      if (data && data.latest_job === null) {
        setLatestJob(null);
      } else {
        setLatestJob(data);
      }
    } catch (err) {
      console.warn('Failed to fetch latest indexing job:', err);
    }
  };

  const fetchJobHistoryData = async () => {
    setLoadingJobHistory(true);
    try {
      const data = await fetchIndexingJobHistory(session.id, 20);
      setJobHistory(data.jobs || []);
    } catch (err) {
      console.warn('Failed to fetch job history:', err);
      setJobHistory([]);
    } finally {
      setLoadingJobHistory(false);
    }
  };

  const handleToggleJobHistory = () => {
    if (!showJobHistory && jobHistory.length === 0) {
      fetchJobHistoryData();
    }
    setShowJobHistory((prev) => !prev);
  };

  const handleCancelIndexing = () => {
    setConfirmCancel(true);
  };

  const doCancelIndexing = async () => {
    setConfirmCancel(false);
    if (isCancelling) return;
    setIsCancelling(true);
    setCancelMessage(null);
    try {
      const result = await cancelLatestIndexingJob(session.id);
      if (result.status === 'no_active_job') {
        setCancelMessage('No active indexing job to cancel.');
      } else {
        setCancelMessage('Cancellation requested. Cleaning up...');
      }
      
      // Give the backend a brief moment to process the cancellation and delete the session data
      await new Promise(r => setTimeout(r, 600));
      
      if (onCancelSession) {
        onCancelSession(session.id);
      } else {
        await fetchLatestJobData();
        await fetchFreshness();
      }
    } catch (err) {
      setCancelMessage(err.message || 'Failed to request cancellation.');
    } finally {
      setIsCancelling(false);
    }
  };

  useEffect(() => {
    setLatestJob(null);
    setCancelMessage(null);
    setJobHistory([]);
    setShowJobHistory(false);
    fetchLatestJobData();
  }, [session.id]);
  const textareaRef = useRef(null);


  const metadataRef = useRef(null);
  const metadataBtnRef = useRef(null);

  const fetchActiveProvider = async () => {
    try {
      const creds = await listProviderCredentials();
      const active = creds.find((c) => c.isActive) || null;
      setActiveProvider(active);
    } catch (err) {
      console.warn('Failed to fetch active provider:', err);
    }
  };

  const fetchFreshness = async () => {
    if (session.status === 'indexing') return;
    try {
      const data = await fetchSessionFreshness(session.id);
      updateSession?.(session.id, { freshness: data });
    } catch (err) {
      console.warn('Failed to fetch freshness:', err);
    }
  };

  const fetchRepoStatus = async () => {
    if (session.status === 'indexing') return;
    setCheckingStatus(true);
    try {
      const data = await fetchSessionRepoStatus(session.id);
      updateSession?.(session.id, { repo_status: data.repo_status });
      await fetchFreshness();
    } catch (err) {
      console.warn('Failed to fetch repo status:', err);
    } finally {
      setCheckingStatus(false);
    }
  };


  useEffect(() => {
    fetchActiveProvider();
    window.addEventListener('CODESEEK_PROVIDER_CHANGED', fetchActiveProvider);
    return () => {
      window.removeEventListener('CODESEEK_PROVIDER_CHANGED', fetchActiveProvider);
    };
  }, [session.id]);

  useEffect(() => {
    fetchRepoStatus();
  }, [session.id, session.status]);

  useEffect(() => {
    let intervalId = null;
    const isCurrentlyIndexing = session.status === 'indexing' || freshnessStatus === 'indexing';
    if (isCurrentlyIndexing && !isCancelling) {
      fetchLatestJobData();
      intervalId = setInterval(() => {
        fetchRepoStatus();
        fetchLatestJobData();
      }, 3000);
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [session.id, session.status, freshnessStatus, isCancelling]);


  useEffect(() => {
    if (!activeProvider) {
      localStorage.removeItem('CODESEEK_ACTIVE_MODEL_OVERRIDE');
      setSelectedModel('');
      return;
    }
    const provider = activeProvider.provider;
    const providerOverride = localStorage.getItem(`CODESEEK_MODEL_OVERRIDE_${provider}`);
    const credentialDefault = activeProvider.model;
    const fallbackDefault = getProviderFallbackModel(provider);

    const resolved = providerOverride || credentialDefault || fallbackDefault;
    setSelectedModel(resolved);
    localStorage.setItem('CODESEEK_ACTIVE_MODEL_OVERRIDE', resolved);
  }, [activeProvider]);

  const handleModelChange = (model) => {
    setSelectedModel(model);
    localStorage.setItem('CODESEEK_ACTIVE_MODEL_OVERRIDE', model);
    if (activeProvider) {
      localStorage.setItem(`CODESEEK_MODEL_OVERRIDE_${activeProvider.provider}`, model);
    }
  };

  useEffect(() => {
    setDismissedFreshnessPrompt(false);
    setShowUpToDatePopup(false);
    setShowSettingsModal(false);
    setShowEvaluation(false);
    setShowMetadata(false);
  }, [session.id]);


  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (
        metadataRef.current &&
        !metadataRef.current.contains(e.target) &&
        metadataBtnRef.current &&
        !metadataBtnRef.current.contains(e.target)
      ) {
        setShowMetadata(false);
      }
      if (
        evaluationRef.current &&
        !evaluationRef.current.contains(e.target) &&
        evaluationBtnRef.current &&
        !evaluationBtnRef.current.contains(e.target)
      ) {
        setShowEvaluation(false);
      }
    };
    window.addEventListener('click', handleOutsideClick);
    return () => window.removeEventListener('click', handleOutsideClick);
  }, []);

  const handleCopySession = () => {
    const messages = activeThread?.messages || [];
    if (messages.length === 0) return;

    const formattedMessages = messages
      .map((msg) => {
        const role = msg.role === 'user' ? 'User' : 'CodeSeek';
        const content = typeof msg.content === 'string' ? msg.content.trim() : '';
        
        let meta = '';
        if (msg.role !== 'user') {
          const modelInfo = selectedModel ? `Model: ${selectedModel}` : '';
          const tokenInfo = msg.context_tokens ? `${msg.context_tokens} tokens` : '';
          const parts = [modelInfo, tokenInfo].filter(Boolean);
          if (parts.length > 0) {
            meta = ` (${parts.join(', ')})`;
          }
        }
        
        let text = `### **${role}**${meta}\n\n${content}`;
        
        if (msg.role !== 'user' && msg.sources && msg.sources.length > 0) {
          const sourceLines = msg.sources
            .map((src) => {
              const file = src.file || src.relative_path || '';
              const symbol = src.symbol || src.symbol_name || '';
              
              let lines = src.lines;
              if (!lines && src.start_line) {
                const start = Number(src.start_line);
                const end = Number(src.end_line);
                if (Number.isFinite(start) && start > 0) {
                  if (Number.isFinite(end) && end > 0 && end !== start) {
                    lines = `${start}-${end}`;
                  } else {
                    lines = String(start);
                  }
                }
              }
              
              return `- ${file}${symbol ? ` :: ${symbol}` : ''}${lines ? ` (lines ${lines})` : ''}`;
            })
            .filter(Boolean);
          if (sourceLines.length > 0) {
            text += `\n\n**Sources:**\n${sourceLines.join('\n')}`;
          }
        }
        
        return text;
      })
      .join('\n\n---\n\n');

    const header = `# CodeSeek Session - ${session.repo_id}\n\n`;
    const fullText = header + formattedMessages;

    navigator.clipboard.writeText(fullText).then(() => {
      setCopiedSession(true);
      setTimeout(() => setCopiedSession(false), 2000);
    });
  };

  const handleIndexLatest = async (force = false) => {
    if (isReindexing) return;
    setIsReindexing(true);
    setReindexingError(null);
    try {
      updateSession?.(session.id, { status: 'indexing' });
      const data = await indexLatestVersion(session.id);
      updateSession?.(session.id, {
        status: data.status || 'indexing',
        freshness: {
          ...session.freshness,
          freshness_status: data.freshness_status || 'indexing',
          can_index_latest: false
        }
      });
      await fetchFreshness();
      await fetchRepoStatus();
    } catch (err) {
      console.error('Failed to trigger index latest:', err);
      setReindexingError(err.message || 'Failed to start indexing.');
      updateSession?.(session.id, { status: 'failed', error: err.message });
    } finally {
      setIsReindexing(false);
    }
  };


  const { isLoading, sendMessage, cancelActiveQuery } = useChat({ appendMessage });
  const isReady = session.status === 'ready';
  const activeThread =
    session.threads?.find((thread) => thread.id === session.active_thread_id) ||
    session.threads?.[0] ||
    null;
  const canChat = isReady && !!activeThread;
  const statusMessage = statusCopy(session);

  // Auto-scroll when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeThread?.messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isLoading || !canChat) return;
    setInput('');
    sendMessage(session, text);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea up to ~3 lines
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 84) + 'px';
  }, [input]);

  const hasMessages = (activeThread?.messages || []).length > 0;

  const repoNamePart = session.repo_full_name ? session.repo_full_name.split('/').pop() : '';
  const isSubdirectorySession = !!(session.repo_root && repoNamePart && !session.repo_root.endsWith(repoNamePart) && !session.repo_root.endsWith(repoNamePart + '/'));

  return (
    <div className="flex flex-col h-full min-w-0 relative">
      {/* Sleek Top Header Bar */}
      <div className="shrink-0 flex items-center justify-between px-6 py-2 bg-surface-2/20 border-b border-border backdrop-blur-md z-10">
        <div className="flex items-center gap-3 min-w-0">
          <span className="font-mono text-sm font-semibold tracking-wide text-text-primary truncate">
            {session.repo_full_name}
          </span>
          {freshnessStatus ? (
            <FreshnessBadge status={freshnessStatus} />
          ) : (
            <FreshnessBadge status={null} />
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {(freshness || repoStatus) && (freshnessStatus !== 'latest' && repoStatus?.status !== 'up_to_date') && (
            <div className="hidden md:flex items-center gap-4 text-text-muted text-[11px] font-mono mr-4 select-none">
              <span>Indexed: <code className="text-text-secondary">{(freshness?.indexed_commit_sha || repoStatus?.indexed_commit_sha)?.slice(0, 7) || 'N/A'}</code></span>
              {(freshnessStatus === 'stale_commit' || repoStatus?.status === 'out_of_date') && (
                <span>Latest: <code className="text-warning">{(freshness?.current_commit_sha || repoStatus?.current_commit_sha)?.slice(0, 7) || 'N/A'}</code></span>
              )}
              {repoStatus?.files_indexed > 0 && (
                <span>Files: <code className="text-text-secondary">{repoStatus.files_indexed}</code></span>
              )}
            </div>
          )}
          
          {hasMessages && (
            <div className="flex items-center gap-1.5 border-r border-border pr-3 mr-1">
              <button
                onClick={handleCopySession}
                title={copiedSession ? "Copied!" : "Copy whole session"}
                className="w-7 h-7 flex items-center justify-center rounded-full bg-surface-3 border border-border text-text-muted hover:text-text-primary hover:border-text-muted transition-all duration-150"
                aria-label="Copy whole session"
              >
                {copiedSession ? <CheckIcon /> : <CopyIcon />}
              </button>
              <button
                onClick={() => setConfirmClear(true)}
                title="Clear chat"
                className="w-7 h-7 flex items-center justify-center rounded-full bg-surface-3 border border-border text-text-muted hover:text-warning hover:border-warning/40 transition-all duration-150"
                aria-label="Clear chat"
              >
                <ClearIcon />
              </button>
            </div>
          )}

          <button
            ref={metadataBtnRef}
            onClick={() => {
              const nextVal = !showMetadata;
              setShowMetadata(nextVal);
            }}
            title="Session metadata & binding info"
            className={`w-7 h-7 flex items-center justify-center rounded-full border transition-all duration-150 mr-1.5 shrink-0 ${
              showMetadata
                ? 'bg-surface-3 border-text-muted text-text-primary'
                : 'bg-surface-3 border-border text-text-muted hover:text-text-primary hover:border-text-muted'
            }`}
            aria-label="Toggle Session Metadata"
          >
            <InfoIcon />
          </button>





        </div>
      </div>

      {/* Collapsible Metadata Panel */}
      {showMetadata && (
        <div ref={metadataRef} className="shrink-0 max-h-[70vh] overflow-y-auto bg-surface-2 border-b border-border px-6 py-4 animate-fadeIn relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs font-mono select-none">
            <div className="space-y-1.5">
              <h4 className="text-[10px] uppercase tracking-wider text-text-muted font-bold">Repository Config</h4>
              <div className="flex flex-col">
                <span className="text-[9px] text-text-muted">Repo Root</span>
                <span className="text-text-primary select-text break-all font-semibold" title={session.repo_root || 'N/A'}>
                  {session.repo_root || 'N/A'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[9px] text-text-muted">Collection</span>
                <span className="text-text-primary select-text break-all font-semibold" title={session.collection || 'N/A'}>
                  {session.collection || 'N/A'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[9px] text-text-muted">Branch</span>
                <span className="text-text-primary font-semibold">{freshness?.current_branch || repoStatus?.current_branch || 'N/A'}</span>
              </div>
            </div>
            
            <div className="space-y-1.5">
              <h4 className="text-[10px] uppercase tracking-wider text-text-muted font-bold">Git Binding</h4>
              <div className="flex flex-col">
                <span className="text-[9px] text-text-muted">Indexed Commit</span>
                <span className="text-text-primary select-text font-semibold" title={freshness?.indexed_commit_sha || repoStatus?.indexed_commit_sha || 'N/A'}>
                  {freshness?.indexed_commit_sha ? freshness.indexed_commit_sha : (repoStatus?.indexed_commit_sha ? repoStatus.indexed_commit_sha : 'N/A')}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[9px] text-text-muted">Current Commit</span>
                <span className={`select-text font-semibold ${(freshness?.current_commit_sha || repoStatus?.current_commit_sha) !== (freshness?.indexed_commit_sha || repoStatus?.indexed_commit_sha) ? 'text-warning' : 'text-text-primary'}`} title={freshness?.current_commit_sha || repoStatus?.current_commit_sha || 'N/A'}>
                  {freshness?.current_commit_sha ? freshness.current_commit_sha : (repoStatus?.current_commit_sha ? repoStatus.current_commit_sha : 'N/A')}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[9px] text-text-muted">Worktree Status</span>
                <span className={(freshness?.worktree_dirty || repoStatus?.dirty_worktree) ? 'text-offline font-bold' : 'text-online font-semibold'}>
                  {(freshness?.worktree_dirty || repoStatus?.dirty_worktree)
                    ? `Dirty (${freshness?.modified_files_count ?? repoStatus?.modified_files_count ?? 0}m, ${freshness?.untracked_files_count ?? repoStatus?.untracked_files_count ?? 0}u, ${freshness?.deleted_files_count ?? repoStatus?.deleted_files_count ?? 0}d)`
                    : 'Clean'}
                </span>
              </div>
            </div>
            
            <div className="space-y-1.5 flex flex-col justify-between">
              <div>
                <h4 className="text-[10px] uppercase tracking-wider text-text-muted font-bold mb-1">Database Stats</h4>
                <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col bg-surface-3 p-1.5 rounded border border-border">
                    <span className="text-[9px] text-text-muted">Files</span>
                    <span className="text-text-primary font-bold">{repoStatus?.files_indexed ?? 0}</span>
                  </div>
                  <div className="flex flex-col bg-surface-3 p-1.5 rounded border border-border">
                    <span className="text-[9px] text-text-muted">Chunks</span>
                    <span className="text-text-primary font-bold">{repoStatus?.chunks_generated ?? 0}</span>
                  </div>
                  <div className="flex flex-col bg-surface-3 p-1.5 rounded border border-border">
                    <span className="text-[9px] text-text-muted">Embeddings</span>
                    <span className="text-text-primary font-bold">{repoStatus?.embeddings_stored ?? 0}</span>
                  </div>
                </div>
                {(repoStatus?.embedding_provider || repoStatus?.embedding_model) && (
                  <div className="flex flex-col bg-surface-3 p-1.5 rounded border border-border mt-2">
                    <span className="text-[9px] text-text-muted">Embeddings</span>
                    <span className="text-text-primary font-bold">
                      {repoStatus?.embedding_provider || 'unknown'} · {repoStatus?.embedding_model || 'unknown'} · {repoStatus?.embedding_dimensions ? `${repoStatus.embedding_dimensions}d` : 'N/A'}
                    </span>
                  </div>
                )}
              </div>
              
              <div className="flex flex-wrap items-center gap-2 mt-3 md:mt-0">
                <button
                  onClick={fetchRepoStatus}
                  disabled={checkingStatus || session.status === 'indexing'}
                  className="flex-1 py-1.5 px-2 text-2xs font-semibold rounded-lg bg-surface-3 border border-border hover:border-text-muted text-text-primary disabled:opacity-40 transition-colors flex items-center justify-center gap-1 min-w-[90px]"
                >
                  <svg className={`w-3.5 h-3.5 ${checkingStatus ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1 1 21.306 7M7 9a5 5 0 0 1 10 0" />
                  </svg>
                  <span>Refresh status</span>
                </button>
                <button
                  onClick={() => handleIndexLatest()}
                  disabled={session.status === 'indexing'}
                  title="Clean repo indexing"
                  className="flex-1 py-1.5 px-2 text-2xs font-semibold rounded-lg bg-text-primary hover:bg-text-secondary text-[#0a0a0a] disabled:opacity-40 transition-colors flex items-center justify-center gap-1 min-w-[90px]"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1 1 21.306 7M7 9a5 5 0 0 1 10 0" />
                  </svg>
                  <span>Index latest</span>
                </button>
              </div>
            </div>
          </div>
          {freshnessStatus && freshnessStatus !== 'indexing' && (
            <div className="mt-4 pt-4 border-t border-border w-full">
              <IndexPreviewPanel
                sessionId={session.id}
                sessionStatus={session.status}
                updateSession={updateSession}
                fetchFreshness={fetchFreshness}
                fetchRepoStatus={fetchRepoStatus}
                freshnessStatus={freshnessStatus}
                canIndexLatest={freshness?.can_index_latest}
                isReindexing={isReindexing}
                sessionFreshness={session.freshness}
              />
            </div>
          )}
        </div>
      )}


      {/* Reindexing Error Notice */}
      {reindexingError && (
        <div className="shrink-0 px-6 pt-3 flex flex-col items-center animate-fadeIn">
          <StatusNotice
            tone="error"
            message={`Reindexing Error: ${reindexingError}`}
            actionLabel="Dismiss"
            onAction={() => setReindexingError(null)}
          />
        </div>
      )}

      {/* Message list or empty state */}
      {!hasMessages ? (
        <div className="flex-1 flex flex-col items-center justify-center pb-16 px-5 min-h-0">
          <IndexingLiveLog session={session} onRetryIndexing={onRetryIndexing} onCancelIndexing={handleCancelIndexing} />
          <EmptyState
            repoName={session.repo_id}
          />
          {/* Input bar inline below empty state */}
          <div className="w-full max-w-xl mt-8">
            <div
              className="flex items-center gap-2 px-4 py-1.5 rounded-2xl border border-border bg-surface-2 shadow-lg transition-colors focus-within:border-text-muted"
              style={{ boxShadow: '0 0 20px rgba(0, 0, 0, 0.5), 0 0 2px rgba(255, 255, 255, 0.03)' }}
            >
              <ModelSelector activeModel={selectedModel} onChange={handleModelChange} activeProvider={activeProvider} />
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading || !canChat}
                placeholder={`Ask about ${session.repo_id}…`}
                rows={1}
                className="flex-1 resize-none bg-transparent border-none text-sm text-text-primary placeholder-text-muted font-sans focus:outline-none disabled:opacity-50 leading-normal"
                style={{ minHeight: '24px', maxHeight: '84px' }}
              />
              <button
                onClick={isLoading ? cancelActiveQuery : handleSend}
                disabled={!isLoading && (!input.trim() || !canChat)}
                title={isLoading ? "Stop generating" : "Send (Enter)"}
                className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-full text-base transition-all duration-150 ${
                  isLoading
                    ? 'bg-offline text-text-primary hover:bg-offline/80'
                    : 'bg-text-primary text-[#0a0a0a] hover:bg-text-secondary disabled:opacity-30 disabled:cursor-not-allowed'
                }`}
                style={isLoading ? undefined : { color: '#0a0a0a' }}
              >
                {isLoading ? <StopIcon /> : <SendIcon />}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="flex-1 overflow-y-auto py-5 min-h-0" style={{ paddingBottom: '100px' }}>
            <div className="w-full space-y-4 px-4 md:px-12">
              <IndexingLiveLog session={session} onRetryIndexing={onRetryIndexing} onCancelIndexing={handleCancelIndexing} />
              {(activeThread?.messages || []).map((msg, idx, arr) => {
                const prevUserQuery = msg.role === 'assistant'
                  ? (arr.slice(0, idx).reverse().find(m => m.role === 'user')?.content || '')
                  : '';
                return (
                  <MessageBubble key={msg.id} message={msg} sessionId={session.id} userQuery={prevUserQuery} />
                );
              })}
              <div ref={bottomRef} />
            </div>
          </div>

          {/* Floating input bar — only when messages exist */}
          <div className="absolute bottom-0 left-0 right-0 px-4 pb-2 pt-6 pointer-events-none"
               style={{ background: 'linear-gradient(to top, #0a0a0a 50%, transparent)' }}>
            <div className="pointer-events-auto max-w-xl mx-auto">
              <div
                className="flex items-center gap-2 px-4 py-1.5 rounded-2xl border border-border bg-surface-2 shadow-lg transition-colors focus-within:border-text-muted"
                style={{ boxShadow: '0 0 20px rgba(0, 0, 0, 0.5), 0 0 2px rgba(255, 255, 255, 0.03)' }}
              >
                <ModelSelector activeModel={selectedModel} onChange={handleModelChange} activeProvider={activeProvider} />
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isLoading || !canChat}
                  placeholder={`Ask about ${session.repo_id}…`}
                  rows={1}
                  className="flex-1 resize-none bg-transparent border-none text-sm text-text-primary placeholder-text-muted font-sans focus:outline-none disabled:opacity-50 leading-normal"
                  style={{ minHeight: '24px', maxHeight: '84px' }}
                />
                <button
                  onClick={isLoading ? cancelActiveQuery : handleSend}
                  disabled={!isLoading && (!input.trim() || !canChat)}
                  title={isLoading ? "Stop generating" : "Send (Enter)"}
                  className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-full text-base transition-all duration-150 ${
                    isLoading
                      ? 'bg-offline text-text-primary hover:bg-offline/80'
                      : 'bg-text-primary text-[#0a0a0a] hover:bg-text-secondary disabled:opacity-30 disabled:cursor-not-allowed'
                  }`}
                  style={isLoading ? undefined : { color: '#0a0a0a' }}
                >
                  {isLoading ? <StopIcon /> : <SendIcon />}
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {confirmClear && (
        <ConfirmDialog
          message="Clear this chat? The repo session will remain available."
          confirmLabel="Clear Chat"
          danger={false}
          onConfirm={() => {
            setConfirmClear(false);
            onClearMessages(session.id);
          }}
          onCancel={() => setConfirmClear(false)}
        />
      )}

      {confirmCancel && (
        <ConfirmDialog
          message="Are you sure you want to cancel indexing? This will completely terminate the job and delete the session."
          confirmLabel="Yes, Cancel & Delete"
          danger={true}
          onConfirm={doCancelIndexing}
          onCancel={() => setConfirmCancel(false)}
        />
      )}

      {freshnessStatus === 'stale_commit' && !dismissedFreshnessPrompt && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fadeIn">
          <div className="w-full max-w-sm bg-surface-2 border border-border rounded-2xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3 text-warning">
              <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <h3 className="text-sm font-semibold text-text-primary font-sans">Repository Out of Date</h3>
            </div>
            <p className="text-xs text-text-secondary leading-relaxed font-sans">
              The indexed commits do not match the latest version available on the remote repository.
            </p>
            <div className="bg-surface-3 p-3 rounded-lg border border-border text-[11px] font-mono space-y-1 text-text-muted">
              <div>Indexed SHA: <span className="text-text-primary">{(freshness?.indexed_commit_sha || repoStatus?.indexed_commit_sha)?.slice(0, 7) || 'N/A'}</span></div>
              <div>Latest SHA: <span className="text-warning">{(freshness?.current_commit_sha || repoStatus?.current_commit_sha)?.slice(0, 7) || 'N/A'}</span></div>
            </div>
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setDismissedFreshnessPrompt(true)}
                className="px-4 py-2 text-xs font-semibold text-text-muted hover:text-text-primary transition-colors"
              >
                Dismiss
              </button>
              <button
                onClick={async () => {
                  setDismissedFreshnessPrompt(true);
                  await handleIndexLatest();
                }}
                disabled={isReindexing || freshness?.can_index_latest === false || freshnessStatus === 'indexing'}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-text-primary text-[#0a0a0a] hover:bg-text-secondary disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {isReindexing ? 'Starting index…' : (freshnessStatus === 'indexing' ? 'Indexing…' : 'Index latest')}
              </button>
            </div>
          </div>
        </div>
      )}

      {showUpToDatePopup && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fadeIn">
          <div className="w-full max-w-sm bg-surface-2 border border-border rounded-2xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3 text-online">
              <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h3 className="text-sm font-semibold text-text-primary font-sans">Latest Repository</h3>
            </div>
            <p className="text-xs text-text-secondary leading-relaxed font-sans">
              The indexed repository is already at the latest version. Re-indexing is not required.
            </p>
            <div className="flex items-center justify-end pt-2">
              <button
                onClick={() => setShowUpToDatePopup(false)}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-text-primary text-[#0a0a0a] hover:bg-text-secondary transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

function statusCopy(session) {
  if (session.status === 'failed') {
    return session.error
      ? `Indexing failed: ${session.error}. Retry indexing after checking GitHub access and backend logs.`
      : 'Indexing failed. Retry after checking GitHub access and backend logs.';
  }
  if (session.status && session.status !== 'ready') {
    return 'Repository indexing is still running. Questions will be enabled when the session becomes ready.';
  }
  return '';
}

function StatusNotice({ tone, message, actionLabel = '', onAction = null, disabled = false }) {
  const toneClass =
    tone === 'error'
      ? 'border-offline/40 bg-offline/10 text-offline'
      : 'border-warning/40 bg-warning/10 text-warning';
  return (
    <div className={`w-full max-w-xl mb-4 rounded-xl border px-4 py-3 text-xs font-mono leading-relaxed ${toneClass}`}>
      <div className="flex items-start justify-between gap-3">
        <div>{message}</div>
        {actionLabel && onAction && (
          <button
            onClick={onAction}
            disabled={disabled}
            className="shrink-0 rounded-full border border-current/30 px-2.5 py-1 text-[10px] uppercase tracking-wide transition-colors hover:bg-black/10 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}

function ClearIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M6.5 1a.5.5 0 0 1 .5.5V2h2v-.5a.5.5 0 0 1 1 0V2h1.5a.5.5 0 0 1 0 1H4.707l6.147 6.146a.5.5 0 0 1 0 .708l-2 2a.5.5 0 0 1-.708 0L2 5.707V11.5A2.5 2.5 0 0 0 4.5 14h5a2.5 2.5 0 0 0 2.5-2.5V8.207a.5.5 0 0 1 1 0V11.5A3.5 3.5 0 0 1 9.5 15h-5A3.5 3.5 0 0 1 1 11.5V4.5a.5.5 0 0 1 .854-.354L8.5 10.793l1.293-1.293L3.146 2.854A.5.5 0 0 1 3.5 2H6v-.5a.5.5 0 0 1 .5-.5z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <rect x="3" y="3" width="10" height="10" rx="1" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      className="animate-spin"
      aria-hidden="true"
    >
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}

const PROVIDER_MODEL_PRESETS = {
  gemini: [
    {
      value: 'gemini-2.0-flash',
      name: 'Gemini 2.0 Flash',
      label: 'Default / Fast load',
      short: '⚡ Flash',
      tooltip: 'Free tier limits: 15 Requests Per Minute (RPM) & 1,000,000 Tokens Per Minute (TPM). High capacity, extremely fast.',
    },
    {
      value: 'gemini-1.5-pro',
      name: 'Gemini 1.5 Pro',
      label: 'Complex queries',
      short: '💎 Pro',
      tooltip: 'Free tier limits: 2 Requests Per Minute (RPM) & 32,000 Tokens Per Minute (TPM). Highly rate-limited; easily triggered on large repositories.',
    },
    {
      value: 'gemini-1.5-flash',
      name: 'Gemini 1.5 Flash',
      label: 'Flash 1.5',
      short: '⚡ Flash (1.5)',
      tooltip: 'Free tier limits: 15 Requests Per Minute (RPM) & 1,000,000 Tokens Per Minute (TPM). Fast and reliable.',
    }
  ],
  groq: [
    {
      value: 'llama-3.3-70b-versatile',
      name: 'Llama 3.3 70B',
      label: 'Default / High quality',
      short: '🦙 Llama 3.3',
      tooltip: 'Versatile 70B model with high rate limits and speed.',
    },
    {
      value: 'llama-3.1-8b-instant',
      name: 'Llama 3.1 8B',
      label: 'Instant replies',
      short: '🦙 Llama 8B',
      tooltip: 'Super fast lightweight model.',
    },
    {
      value: 'mixtral-8x7b-32768',
      name: 'Mixtral 8x7B',
      label: 'Mixtral Mixture of Experts',
      short: '🌀 Mixtral',
      tooltip: 'Good general reasoning model.',
    }
  ],
  openai: [
    {
      value: 'gpt-4o-mini',
      name: 'GPT-4o Mini',
      label: 'Default / Fast & cheap',
      short: '✨ 4o Mini',
      tooltip: 'Very fast, cost-effective model.',
    },
    {
      value: 'gpt-4o',
      name: 'GPT-4o',
      label: 'Advanced intelligence',
      short: '🧠 GPT-4o',
      tooltip: 'High intelligence, premium general model.',
    },
    {
      value: 'gpt-3.5-turbo',
      name: 'GPT-3.5 Turbo',
      label: 'Legacy Fast',
      short: '⚡ GPT-3.5',
      tooltip: 'Standard fast model.',
    }
  ],
  openrouter: [
    {
      value: 'google/gemini-2.0-flash',
      name: 'Gemini 2.0 Flash',
      label: 'Gemini 2.0 Flash via OpenRouter',
      short: '⚡ Gemini',
      tooltip: 'Fast and efficient Gemini 2.0 Flash.',
    },
    {
      value: 'openai/gpt-4o-mini',
      name: 'GPT-4o Mini',
      label: 'GPT-4o Mini via OpenRouter',
      short: '✨ 4o Mini',
      tooltip: 'High-speed, low-cost intelligence.',
    },
    {
      value: 'meta-llama/llama-3-8b-instruct',
      name: 'Llama 3 8B',
      label: 'Llama 3 8B Instruct via OpenRouter',
      short: '🦙 Llama 3',
      tooltip: 'Open-source instruction-tuned model.',
    }
  ],
  aicredits: [
    {
      value: 'gpt-5.4-mini',
      name: 'GPT-5.4 Mini',
      label: 'Default / AI Credits',
      short: '🪙 GPT-5.4',
      tooltip: 'GPT-5.4 Mini via AI Credits. Fast and cost-effective.',
    },
    {
      value: 'deepseek/deepseek-v4-flash',
      name: 'DeepSeek v4 Flash',
      label: 'DeepSeek v4 Flash via AI Credits',
      short: '🐳 DeepSeek v4',
      tooltip: 'DeepSeek v4 Flash via AI Credits. Ultra-fast and cost-effective.',
    },
  ],
};

function ModelSelector({ activeModel, onChange, activeProvider }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    window.addEventListener('click', handleOutsideClick);
    return () => window.removeEventListener('click', handleOutsideClick);
  }, []);

  if (!activeProvider) {
    return (
      <div className="relative shrink-0 flex items-center">
        <button
          type="button"
          title="No active LLM provider configured. Click to configure API tokens."
          onClick={() => {
            window.dispatchEvent(new Event('CODESEEK_OPEN_API_MODAL'));
          }}
          className="flex items-center gap-1.5 rounded-lg border border-warning/40 bg-warning/10 px-2 py-1 text-2xs font-mono font-medium text-warning hover:bg-warning/20 transition-colors select-none animate-pulse"
        >
          <span>⚠️ Setup API</span>
        </button>
      </div>
    );
  }

  const provider = activeProvider.provider;
  const presets = PROVIDER_MODEL_PRESETS[provider] || [];

  const isPreset = presets.some((p) => p.value === activeModel);
  const current = isPreset
    ? presets.find((p) => p.value === activeModel)
    : {
        value: activeModel,
        name: activeModel || 'Default Model',
        label: 'Active Model',
        short: activeModel
          ? activeModel.length > 12
            ? activeModel.substring(0, 10) + '…'
            : activeModel
          : 'Default',
        tooltip: `Active model: ${activeModel || 'Default model'}`,
      };

  return (
    <div className="relative shrink-0 flex items-center" ref={dropdownRef}>
      <button
        type="button"
        title={`Active model: ${current.name}. Click to switch.`}
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="flex items-center gap-1.5 rounded-lg border border-border bg-surface-3 px-2 py-1 text-2xs font-mono font-medium text-text-secondary hover:text-text-primary hover:border-text-muted transition-colors select-none"
      >
        <span>{current.short}</span>
        <svg
          className={`h-2.5 w-2.5 transform text-text-muted transition-transform duration-150 ${
            isOpen ? 'rotate-180' : ''
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={3}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div
          className="absolute bottom-full left-0 mb-2 w-52 rounded-xl border border-border bg-surface-2 p-1 shadow-xl animate-fadeIn z-30 flex flex-col"
          style={{ boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)' }}
        >
          <div className="max-h-48 overflow-y-auto">
            {presets.map((opt) => (
              <button
                key={opt.value}
                type="button"
                title={opt.tooltip}
                onClick={() => {
                  onChange(opt.value);
                  setIsOpen(false);
                }}
                className={`w-full text-left rounded-lg px-2.5 py-1.5 hover:bg-surface-3 transition-colors flex flex-col ${
                  opt.value === activeModel ? 'bg-surface-3/50' : ''
                }`}
              >
                <span className="text-2xs font-medium text-text-primary">{opt.name}</span>
                <span className="text-[10px] text-text-muted">{opt.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CopyIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z" />
      <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" className="text-online" aria-hidden="true">
      <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z" />
    </svg>
  );
}

function FreshnessBadge({ status }) {
  if (!status) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-border bg-surface-2 text-text-muted select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-text-muted/50 animate-pulse" />
        Checking...
      </span>
    );
  }
  
  if (status === 'latest' || status === 'up_to_date') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-online/30 bg-online/10 text-online select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-online" />
        Indexed to latest
      </span>
    );
  }
  
  if (status === 'stale_commit' || status === 'out_of_date') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-warning/30 bg-warning/10 text-warning select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />
        Repo has new commits
      </span>
    );
  }

  if (status === 'branch_changed') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-warning/30 bg-warning/10 text-warning select-none" title="Active branch differs from indexed branch">
        <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />
        Branch changed
      </span>
    );
  }
  
  if (status === 'dirty_worktree') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-offline/30 bg-offline/10 text-offline select-none" title="Uncommitted changes in local workspace">
        <span className="w-1.5 h-1.5 rounded-full bg-offline" />
        Uncommitted changes
      </span>
    );
  }
  
  if (status === 'indexing') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-online/30 bg-online/10 text-online select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-online animate-pulse" />
        Indexing
      </span>
    );
  }

  if (status === 'stale_indexing') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-warning/30 bg-warning/10 text-warning select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />
        Indexing appears stuck
      </span>
    );
  }

  if (status === 'failed') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-offline/30 bg-offline/10 text-offline select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-offline" />
        Indexing failed
      </span>
    );
  }
  
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium font-mono border border-border bg-surface-2 text-text-muted select-none">
      <span className="w-1.5 h-1.5 rounded-full bg-text-muted/50" />
      Freshness unknown
    </span>
  );
}

function ThreeDotsIcon() {
  return (
    <svg width="4" height="16" viewBox="0 0 4 16" fill="currentColor" aria-hidden="true">
      <path d="M2 10a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0-6a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 12a2 2 0 1 1 0-4 2 2 0 0 1 0 4z" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function FileGroupSection({
  title,
  files,
  colorClass,
  isExpanded,
  onToggleExpand,
  showAll,
  onToggleShowAll,
  prefix,
}) {
  if (!files || files.length === 0) return null;
  const visibleFiles = showAll ? files : files.slice(0, 5);
  const hasMore = files.length > 5;

  return (
    <div className="border border-border rounded-lg bg-surface-1 overflow-hidden mb-2">
      <button
        type="button"
        onClick={onToggleExpand}
        className="w-full flex items-center justify-between p-2.5 bg-surface-2 hover:bg-surface-3 transition-colors text-left"
      >
        <span className={`font-semibold tracking-wide uppercase text-[9px] ${colorClass}`}>
          {title} ({files.length})
        </span>
        <svg
          className={`w-3.5 h-3.5 text-text-muted transition-transform duration-200 ${isExpanded ? 'rotate-90' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {isExpanded && (
        <div className="p-2.5 space-y-1 text-[10px] divide-y divide-border/20 border-t border-border/50 max-h-[250px] overflow-y-auto bg-surface-1">
          {visibleFiles.map((file) => (
            <div key={file} className="py-1 flex items-center justify-between text-text-secondary select-text font-mono truncate" title={file}>
              <span className="truncate">{prefix} {file}</span>
            </div>
          ))}
          {hasMore && (
            <button
              type="button"
              onClick={onToggleShowAll}
              className="w-full text-center text-text-secondary hover:text-text-primary py-1.5 mt-1 border-t border-dashed border-border/30 text-[9px] uppercase tracking-wider font-semibold transition-colors"
            >
              {showAll ? 'Show less' : `Show ${files.length - 5} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function IndexPreviewPanel({
  sessionId,
  sessionStatus,
  updateSession,
  fetchFreshness,
  fetchRepoStatus,
  freshnessStatus,
  canIndexLatest,
  isReindexing,
  sessionFreshness
}) {
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  const [isIncrementalDisabled, setIsIncrementalDisabled] = useState(false);
  const [incrementalError, setIncrementalError] = useState(null);
  const [incrementalSuccess, setIncrementalSuccess] = useState(null);
  const [isTriggering, setIsTriggering] = useState(false);

  const [showAllModified, setShowAllModified] = useState(false);
  const [showAllAdded, setShowAllAdded] = useState(false);
  const [showAllDeleted, setShowAllDeleted] = useState(false);
  const [showAllUnchanged, setShowAllUnchanged] = useState(false);

  const [isModifiedExpanded, setIsModifiedExpanded] = useState(true);
  const [isAddedExpanded, setIsAddedExpanded] = useState(true);
  const [isDeletedExpanded, setIsDeletedExpanded] = useState(true);
  const [isUnchangedExpanded, setIsUnchangedExpanded] = useState(false);

  useEffect(() => {
    async function loadPreview() {
      setLoading(true);
      setError(null);
      setIncrementalError(null);
      setIncrementalSuccess(null);
      try {
        const data = await fetchIndexPreview(sessionId);
        setPreview(data);
      } catch (err) {
        setError(err.message || 'Failed to fetch index preview.');
      } finally {
        setLoading(false);
      }
    }
    loadPreview();
  }, [sessionId]);

  const handleIndexIncremental = async () => {
    if (isTriggering || isReindexing || sessionStatus === 'indexing') return;
    setIsTriggering(true);
    setIncrementalError(null);
    setIncrementalSuccess(null);
    try {
      const data = await indexSessionIncremental(sessionId);
      if (data.status === 'ready') {
        setIncrementalSuccess(data.message || 'No indexing required: 0 changed files.');
        await fetchFreshness?.();
        await fetchRepoStatus?.();
        const previewData = await fetchIndexPreview(sessionId);
        setPreview(previewData);
      } else {
        updateSession?.(sessionId, {
          status: data.status || 'indexing',
          freshness: {
            ...sessionFreshness,
            freshness_status: data.freshness_status || 'indexing',
            can_index_latest: false
          }
        });
        await fetchFreshness?.();
        await fetchRepoStatus?.();
      }
    } catch (err) {
      console.error('Failed to trigger incremental index:', err);
      if (err.message.includes('not enabled on this server') || err.message.includes('403')) {
        setIsIncrementalDisabled(true);
      }
      setIncrementalError(err.message || 'Incremental indexing failed to start.');
    } finally {
      setIsTriggering(false);
    }
  };

  const isSessionIndexing = sessionStatus === 'indexing' || freshnessStatus === 'indexing' || isReindexing;
  const canIndex = canIndexLatest !== false;

  const modifiedFiles = preview?.changed_files || [];
  const addedFiles = preview?.added_files || [];
  const deletedFiles = preview?.deleted_files || [];
  const unchangedFiles = preview?.unchanged_files || [];

  const isButtonEnabled =
    preview &&
    preview.can_incremental_reindex === true &&
    !isSessionIndexing;

  let blockMessage = null;
  let blockType = '';

  const blockReason = preview?.incremental_block_reason || '';

  if (isIncrementalDisabled || blockReason === 'feature_disabled') {
    blockMessage = "Incremental indexing is not enabled on this server.";
    blockType = 'error';
  } else if (blockReason === 'branch_changed') {
    blockMessage = "Incremental indexing is blocked because the active branch differs from the indexed branch. Use Index latest to switch branches.";
    blockType = 'error';
  } else if (blockReason === 'metadata_unavailable') {
    blockMessage = "Incremental metadata is unavailable. Use Index latest first.";
    blockType = 'warning';
  } else if (blockReason === 'session_failed') {
    blockMessage = "This session is in a failed state. Use Index latest to recover before running incremental indexing.";
    blockType = 'error';
  } else if (blockReason === 'active_indexing') {
    blockMessage = "Indexing is already running.";
    blockType = 'info';
  } else if (blockReason === 'no_changes') {
    blockMessage = "No changed files detected.";
    blockType = 'info';
  } else if (blockReason === 'unknown') {
    blockMessage = "Incremental indexing is unavailable for the current session state. Use Index latest as fallback.";
    blockType = 'warning';
  }

  return (
    <div className="w-full flex flex-col gap-3 font-mono text-xs select-none">
      {loading && (
        <div className="flex items-center justify-center py-6 text-text-muted">
          <span className="animate-pulse">Loading preview data...</span>
        </div>
      )}

      {error && (
        <div className="text-offline bg-offline/10 border border-offline/20 rounded-lg p-3">
          {error}
        </div>
      )}

      {!loading && !error && preview && (
        <div className="flex flex-col gap-4">
          {/* Summary Cards Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="bg-surface-2 border border-border p-2.5 rounded-lg text-center flex flex-col justify-center">
              <span className="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Modified</span>
              <span className="text-sm font-semibold text-warning">{modifiedFiles.length}</span>
            </div>
            <div className="bg-surface-2 border border-border p-2.5 rounded-lg text-center flex flex-col justify-center">
              <span className="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Added</span>
              <span className="text-sm font-semibold text-online">{addedFiles.length}</span>
            </div>
            <div className="bg-surface-2 border border-border p-2.5 rounded-lg text-center flex flex-col justify-center">
              <span className="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Deleted</span>
              <span className="text-sm font-semibold text-offline">{deletedFiles.length}</span>
            </div>
            <div className="bg-surface-2 border border-border p-2.5 rounded-lg text-center flex flex-col justify-center">
              <span className="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Unchanged</span>
              <span className="text-sm font-semibold text-text-secondary">{unchangedFiles.length}</span>
            </div>
            <div className="bg-surface-2 border border-border p-2.5 rounded-lg text-center flex flex-col justify-center col-span-2 sm:col-span-1">
              <span className="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Total Updates</span>
              <span className="text-sm font-semibold text-text-primary">{preview.estimated_files_to_update || 0}</span>
            </div>
          </div>

          {/* Block/Friendly message box if present */}
          {blockMessage && (
            <div className={`p-3 rounded-lg flex items-start gap-2 border text-[10px] ${
              blockType === 'error'
                ? 'bg-offline/10 border-offline/20 text-offline'
                : blockType === 'warning'
                ? 'bg-warning/10 border-warning/20 text-warning'
                : 'bg-surface-2 border-border text-text-muted'
            }`}>
              <InfoIcon />
              <span>{blockMessage}</span>
            </div>
          )}

          {/* Main layout: left side files list, right side actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left Area: Files List (takes 2 cols on md) */}
            <div className="md:col-span-2 flex flex-col gap-2 max-h-[350px] overflow-y-auto pr-1">
              <FileGroupSection
                title="Modified Files"
                files={modifiedFiles}
                colorClass="text-warning"
                isExpanded={isModifiedExpanded}
                onToggleExpand={() => setIsModifiedExpanded(!isModifiedExpanded)}
                showAll={showAllModified}
                onToggleShowAll={() => setShowAllModified(!showAllModified)}
                prefix="~"
              />
              <FileGroupSection
                title="Added Files"
                files={addedFiles}
                colorClass="text-online"
                isExpanded={isAddedExpanded}
                onToggleExpand={() => setIsAddedExpanded(!isAddedExpanded)}
                showAll={showAllAdded}
                onToggleShowAll={() => setShowAllAdded(!showAllAdded)}
                prefix="+"
              />
              <FileGroupSection
                title="Deleted Files"
                files={deletedFiles}
                colorClass="text-offline"
                isExpanded={isDeletedExpanded}
                onToggleExpand={() => setIsDeletedExpanded(!isDeletedExpanded)}
                showAll={showAllDeleted}
                onToggleShowAll={() => setShowAllDeleted(!showAllDeleted)}
                prefix="-"
              />
              <FileGroupSection
                title="Unchanged Files"
                files={unchangedFiles}
                colorClass="text-text-muted"
                isExpanded={isUnchangedExpanded}
                onToggleExpand={() => setIsUnchangedExpanded(!isUnchangedExpanded)}
                showAll={showAllUnchanged}
                onToggleShowAll={() => setShowAllUnchanged(!showAllUnchanged)}
                prefix="•"
              />
            </div>

            {/* Right Area: Buttons & Actions (takes 1 col on md) */}
            <div className="flex flex-col justify-between gap-4 md:border-l border-border md:pl-6">
              <div className="flex flex-col gap-3">
                <div className="text-[9px] uppercase tracking-wider text-text-muted font-bold">Actions</div>
                
                <div className="flex flex-col gap-2">
                  <button
                    type="button"
                    onClick={handleIndexIncremental}
                    disabled={!isButtonEnabled || isTriggering}
                    className={`px-3 py-2 rounded-lg font-mono text-[10px] uppercase tracking-wider font-semibold transition-all flex items-center justify-center gap-1.5 ${
                      isButtonEnabled && !isTriggering
                        ? 'bg-warning/20 border border-warning/40 hover:bg-warning/30 text-warning shadow-md shadow-warning/5 cursor-pointer font-bold'
                        : 'bg-surface-2 border border-border text-text-muted cursor-not-allowed opacity-50'
                    }`}
                  >
                    {isTriggering ? 'Triggering...' : 'Index changed files'}
                  </button>

                  <div className="flex items-center gap-1.5">
                    <span className="bg-warning/10 border border-warning/30 text-warning px-1.5 py-0.5 rounded text-[8px] font-bold tracking-widest uppercase">
                      Experimental
                    </span>
                    <span className="text-[9px] text-text-muted italic">
                      Incremental Flow (V1)
                    </span>
                  </div>
                </div>

                {/* Safety Note */}
                <div className="flex items-start gap-1.5 bg-surface-2 p-2.5 rounded-lg border border-border text-[9px] text-text-muted leading-relaxed">
                  <InfoIcon />
                  <span>Unchanged files are preserved. Only added, modified, and deleted files are processed.</span>
                </div>
              </div>

              <div className="space-y-2">
                {/* Action Meanings */}
                <div className="bg-surface-2/50 border border-border/50 rounded-lg p-2.5 space-y-1.5 text-[9px] text-text-muted">
                  <div>
                    <strong className="text-text-primary">Index changed files:</strong> updates only added, modified, and deleted files.
                  </div>
                  <div>
                    <strong className="text-text-primary">Index latest:</strong> full clean reindex (fallback).
                  </div>
                </div>

                {incrementalError && (
                  <div className="text-offline bg-offline/10 border border-offline/20 rounded-lg p-2 text-[10px]">
                    {incrementalError}
                  </div>
                )}

                {incrementalSuccess && (
                  <div className="text-online bg-online/10 border border-online/20 rounded-lg p-2 text-[10px]">
                    {incrementalSuccess}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
