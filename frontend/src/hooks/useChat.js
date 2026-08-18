import { useState, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { querySessionStream } from '../utils/api.js';

export function useChat({ appendMessage }) {
  const [isLoading, setIsLoading] = useState(false);
  const pendingSessionId = useRef(null);
  const abortControllerRef = useRef(null);
  const activeRequestIdRef = useRef(null);

  const cancelActiveQuery = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
    pendingSessionId.current = null;
    activeRequestIdRef.current = null;
  }, []);

  const sendMessage = useCallback(
    async (session, questionText, options = {}) => {
      if (isLoading || !questionText.trim()) return;
      if (session.status && session.status !== 'ready') return;
      if (!session?.id) {
        throw new Error('Cannot query without a session id.');
      }

      const activeThreadId = session.active_thread_id || session.threads?.[0]?.id || '';
      if (!activeThreadId) {
        throw new Error('Conversation thread is still loading. Try again in a moment.');
      }
      const trimmed = questionText.trim();
      setIsLoading(true);
      pendingSessionId.current = session.id;

      // 1. Append user message immediately
      const userMessage = {
        id: uuidv4(),
        role: 'user',
        content: trimmed,
        sources: [],
        timestamp: new Date().toISOString(),
        error: false,
      };
      appendMessage(session.id, activeThreadId, userMessage);

      // 2. Append loading placeholder
      const loadingId = uuidv4();
      const loadingMessage = {
        id: loadingId,
        role: 'assistant',
        content: null,
        sources: [],
        timestamp: new Date().toISOString(),
        loading: true,
        error: false,
      };
      appendMessage(session.id, activeThreadId, loadingMessage);

      const controller = new AbortController();
      abortControllerRef.current = controller;
      const requestId = uuidv4();
      activeRequestIdRef.current = requestId;
      const isCurrentRequest = () => activeRequestIdRef.current === requestId;

      // Drip buffer: accumulate incoming deltas and release them gradually
      let dripBuffer = '';
      let dripTimer = null;
      const DRIP_INTERVAL_MS = 18;
      const DRIP_CHARS = 4;

      const flushDrip = () => {
        if (dripBuffer.length === 0) {
          dripTimer = null;
          return;
        }
        const chunk = dripBuffer.slice(0, DRIP_CHARS);
        dripBuffer = dripBuffer.slice(DRIP_CHARS);
        accumulatedAnswer += chunk;
        appendMessage(session.id, activeThreadId, {
          __replaceId: loadingId,
          id: loadingId,
          role: 'assistant',
          content: accumulatedAnswer,
          sources: answerSources,
          diagnostics: answerDiagnostics,
          context_tokens: contextTokens,
          timestamp: new Date().toISOString(),
          loading: true,
          error: false,
        });
        dripTimer = setTimeout(flushDrip, DRIP_INTERVAL_MS);
      };

      const scheduleDrip = (text) => {
        if (authoritativeAnswer !== null) return;
        dripBuffer += text;
        if (!dripTimer) {
          dripTimer = setTimeout(flushDrip, DRIP_INTERVAL_MS);
        }
      };

      // Flush remaining buffer when stream ends
      const flushRemainingDrip = () => {
        if (dripTimer) {
          clearTimeout(dripTimer);
          dripTimer = null;
        }
        if (authoritativeAnswer === null && dripBuffer.length > 0) {
          accumulatedAnswer += dripBuffer;
        }
        dripBuffer = '';
      };

      let accumulatedAnswer = '';
      let answerSources = [];
      let answerDiagnostics = null;
      let contextTokens = null;
      let authoritativeAnswer = null;
      let generationStatus = null;
      let streamFailure = null;

      let finalMessageId = loadingId;
      const renderInterrupted = (message = 'Generation stopped.') => appendMessage(
        session.id,
        activeThreadId,
        {
          __replaceId: loadingId,
          id: authoritativeAnswer === null ? loadingId : finalMessageId,
          role: 'assistant',
          content: authoritativeAnswer ?? message,
          sources: authoritativeAnswer === null ? [] : answerSources,
          diagnostics: authoritativeAnswer === null ? null : answerDiagnostics,
          context_tokens: authoritativeAnswer === null ? null : contextTokens,
          timestamp: new Date().toISOString(),
          loading: false,
          error: true,
          generation_status: authoritativeAnswer === null ? 'cancelled' : generationStatus,
        },
      );

      try {
        await querySessionStream({
          question: trimmed,
          session_id: session.id,
          thread_id: activeThreadId,
          graph_retrieval_mode: options.graphRetrievalMode || 'standard',
          request_id: requestId,
          signal: controller.signal,
          onStatus: (status) => {
            if (!isCurrentRequest()) return;
            console.log('[useChat] Status:', status);
          },
          onDelta: (text) => {
            if (!isCurrentRequest()) return;
            scheduleDrip(text);
          },
          onFinalAnswer: (event) => {
            if (!isCurrentRequest()) return;
            if (dripTimer) clearTimeout(dripTimer);
            dripTimer = null;
            dripBuffer = '';
            authoritativeAnswer = event.text;
            accumulatedAnswer = event.text;
            finalMessageId = event.message_id;
            generationStatus = event.generation_status;
            appendMessage(session.id, activeThreadId, {
              __replaceId: loadingId,
              id: loadingId,
              role: 'assistant',
              content: authoritativeAnswer,
              sources: answerSources,
              diagnostics: answerDiagnostics,
              context_tokens: contextTokens,
              generation_status: generationStatus,
              timestamp: new Date().toISOString(),
              loading: true,
              error: false,
            });
          },
          onSources: (data) => {
            if (!isCurrentRequest()) return;
            answerSources = data.sources || [];
            answerDiagnostics = data.diagnostics || null;
            contextTokens = data.context_tokens;
            
            appendMessage(session.id, activeThreadId, {
              __replaceId: loadingId,
              id: loadingId,
              role: 'assistant',
              content: accumulatedAnswer,
              sources: answerSources,
              diagnostics: answerDiagnostics,
              context_tokens: contextTokens,
              timestamp: new Date().toISOString(),
              loading: true,
              error: false,
            });
          },
          onDone: (event) => {
            if (!isCurrentRequest()) return;
            console.log('[useChat] Stream done.');
            generationStatus = event?.status || generationStatus;
            if (event && event.message_id) {
              finalMessageId = event.message_id;
            }
            if (event?.status === 'cancelled' && !streamFailure) {
              streamFailure = new Error('Generation was cancelled.');
            }
          },
          onError: (errMsg) => {
            if (!isCurrentRequest()) return;
            streamFailure = new Error(errMsg);
          },
        });

        // Flush any remaining buffered text
        flushRemainingDrip();

        if (controller.signal.aborted) {
          renderInterrupted();
          return;
        }
        if (!isCurrentRequest()) return;
        if (streamFailure) throw streamFailure;

        const assistantMessage = {
          id: finalMessageId,
          role: 'assistant',
          content: authoritativeAnswer ?? accumulatedAnswer,
          sources: answerSources,
          diagnostics: answerDiagnostics,
          context_tokens: contextTokens,
          timestamp: new Date().toISOString(),
          loading: false,
          error: false,
          generation_status: generationStatus || 'complete',
        };
        appendMessage(session.id, activeThreadId, { __replaceId: loadingId, ...assistantMessage });

      } catch (err) {
        flushRemainingDrip();
        if (controller.signal.aborted) {
          renderInterrupted();
          return;
        }

        console.error('[useChat] Query failed:', err);
        const errorMessage = {
          id: authoritativeAnswer === null ? loadingId : finalMessageId,
          role: 'assistant',
          content: authoritativeAnswer ?? err.message ?? 'Something went wrong. Please try again.',
          sources: authoritativeAnswer === null ? [] : answerSources,
          diagnostics: authoritativeAnswer === null ? null : answerDiagnostics,
          context_tokens: authoritativeAnswer === null ? null : contextTokens,
          timestamp: new Date().toISOString(),
          loading: false,
          error: true,
          generation_status: generationStatus,
        };
        appendMessage(session.id, activeThreadId, { __replaceId: loadingId, ...errorMessage });
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
        if (activeRequestIdRef.current === requestId) {
          activeRequestIdRef.current = null;
          pendingSessionId.current = null;
          setIsLoading(false);
        }
      }
    },
    [isLoading, appendMessage]
  );

  return { isLoading, sendMessage, cancelActiveQuery };
}
