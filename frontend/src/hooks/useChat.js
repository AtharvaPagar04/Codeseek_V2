import { useState, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { querySessionStream } from '../utils/api';

export function useChat({ appendMessage }) {
  const [isLoading, setIsLoading] = useState(false);
  const pendingSessionId = useRef(null);
  const abortControllerRef = useRef(null);

  const cancelActiveQuery = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
    pendingSessionId.current = null;
  }, []);

  const sendMessage = useCallback(
    async (session, questionText) => {
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
        if (dripBuffer.length > 0) {
          accumulatedAnswer += dripBuffer;
          dripBuffer = '';
        }
      };

      let accumulatedAnswer = '';
      let answerSources = [];
      let answerDiagnostics = null;
      let contextTokens = null;

      let finalMessageId = loadingId;

      try {
        await querySessionStream({
          question: trimmed,
          session_id: session.id,
          thread_id: activeThreadId,
          signal: controller.signal,
          onStatus: (status) => {
            console.log('[useChat] Status:', status);
          },
          onDelta: (text) => {
            scheduleDrip(text);
          },
          onSources: (data) => {
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
            console.log('[useChat] Stream done.');
            if (event && event.message_id) {
              finalMessageId = event.message_id;
            }
          },
          onError: (errMsg) => {
            throw new Error(errMsg);
          },
        });

        // Flush any remaining buffered text
        flushRemainingDrip();

        if (controller.signal.aborted) {
          return;
        }

        const assistantMessage = {
          id: finalMessageId,
          role: 'assistant',
          content: accumulatedAnswer || '(no answer returned)',
          sources: answerSources,
          diagnostics: answerDiagnostics,
          context_tokens: contextTokens,
          timestamp: new Date().toISOString(),
          loading: false,
          error: false,
        };
        appendMessage(session.id, activeThreadId, { __replaceId: loadingId, ...assistantMessage });

      } catch (err) {
        flushRemainingDrip();
        if (controller.signal.aborted) {
          const assistantMessage = {
            id: loadingId,
            role: 'assistant',
            content: accumulatedAnswer || 'Generation stopped.',
            sources: answerSources,
            diagnostics: answerDiagnostics,
            context_tokens: contextTokens,
            timestamp: new Date().toISOString(),
            loading: false,
            error: false,
          };
          appendMessage(session.id, activeThreadId, { __replaceId: loadingId, ...assistantMessage });
          return;
        }

        console.error('[useChat] Query failed:', err);
        const errorMessage = {
          id: loadingId,
          role: 'assistant',
          content: err.message || 'Something went wrong. Please try again.',
          sources: [],
          timestamp: new Date().toISOString(),
          loading: false,
          error: true,
        };
        appendMessage(session.id, activeThreadId, { __replaceId: loadingId, ...errorMessage });
      } finally {
        setIsLoading(false);
        pendingSessionId.current = null;
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }
    },
    [isLoading, appendMessage]
  );

  return { isLoading, sendMessage, cancelActiveQuery };
}
