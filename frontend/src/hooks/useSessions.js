import { useState, useCallback } from 'react';

export const sortByLastActive = (sessions) =>
  [...sessions].sort(
    (a, b) =>
      new Date(b.last_active || b.created_at) - new Date(a.last_active || a.created_at)
  );

export const normalizeThreads = (threads) =>
  Array.isArray(threads) ? threads.map((thread) => ({ ...thread, messages: thread.messages || [] })) : [];

export const normalizeSessionRecord = (sessionData, current = null, options = {}) => {
  const now = options.now || new Date().toISOString();
  const repoFullName = sessionData.repo_full_name || current?.repo_full_name || '';
  const repoId = repoFullName.split('/').pop() || sessionData.repo_id || current?.repo_id || 'repository';
  const createdAt = sessionData.created_at || current?.created_at || now;
  return {
    ...current,
    id: sessionData.id,
    repo_id: repoId,
    repo_full_name: repoFullName,
    repo_description: sessionData.repo_description || current?.repo_description || '',
    repo_private: sessionData.repo_private ?? current?.repo_private ?? false,
    status: sessionData.status || current?.status || 'indexing',
    error: sessionData.error || '',
    created_at: createdAt,
    last_active: options.lastActive || current?.last_active || sessionData.updated_at || createdAt,
    threads: current?.threads || [],
    active_thread_id: current?.active_thread_id || current?.threads?.[0]?.id || null,
    repo_status: sessionData.repo_status || current?.repo_status || null,
  };
};

export const applyClearSessionMessages = (sessions, sessionId, now) => {
  const next = sessions.map((session) =>
    session.id === sessionId
      ? {
          ...session,
          threads: session.threads.map((thread) =>
            thread.id === session.active_thread_id ? { ...thread, messages: [] } : thread
          ),
          last_active: now,
        }
      : session
  );
  return sortByLastActive(next);
};

export const applySetThreadMessages = (sessions, sessionId, threadId, messages) => {
  const next = sessions.map((session) =>
    session.id === sessionId
      ? {
          ...session,
          threads: session.threads.map((thread) =>
            thread.id === threadId
              ? { ...thread, messages: Array.isArray(messages) ? messages : [] }
              : thread
          ),
        }
      : session
  );
  return sortByLastActive(next);
};

export const applyAppendMessage = (sessions, sessionId, threadId, message, now) => {
  const next = sessions.map((session) => {
    if (session.id !== sessionId) return session;

    const threads = session.threads.map((thread) => {
      if (thread.id !== threadId) return thread;
      let messages;
      if (message.__replaceId) {
        const { __replaceId, ...realMessage } = message;
        messages = (thread.messages || []).map((m) => (m.id === __replaceId ? realMessage : m));
      } else {
        messages = [...(thread.messages || []), message];
      }
      return { ...thread, messages };
    });

    return { ...session, last_active: now, threads };
  });
  return sortByLastActive(next);
};

export const applySetSessionThreads = (sessions, sessionId, threads) => {
  const next = sessions.map((session) =>
    session.id === sessionId
      ? (() => {
          const normalizedThreads = normalizeThreads(threads);
          const activeThreadId = normalizedThreads.some((thread) => thread.id === session.active_thread_id)
            ? session.active_thread_id
            : normalizedThreads[0]?.id || null;
          return {
            ...session,
            threads: normalizedThreads,
            active_thread_id: activeThreadId,
          };
        })()
      : session
  );
  return sortByLastActive(next);
};

export const applyDeleteSession = (sessions, sessionId) =>
  sortByLastActive(sessions.filter((s) => s.id !== sessionId));



export function useSessions() {
  const [sessions, setSessions] = useState([]);

  const addSession = useCallback((sessionData) => {
    const now = new Date().toISOString();
    const newSession = normalizeSessionRecord(sessionData, sessions.find((s) => s.id === sessionData.id) || null, {
      now,
      lastActive: now,
    });
    setSessions((prev) => {
      const current = prev.find((s) => s.id === sessionData.id) || null;
      const merged = normalizeSessionRecord(sessionData, current, { now, lastActive: now });
      return sortByLastActive([merged, ...prev.filter((s) => s.id !== merged.id)]);
    });
    return newSession;
  }, [sessions]);

  const deleteSession = useCallback((sessionId) => {
    setSessions((prev) => sortByLastActive(prev.filter((s) => s.id !== sessionId)));
  }, []);

  const clearSessionMessages = useCallback((sessionId) => {
    const now = new Date().toISOString();
    setSessions((prev) => applyClearSessionMessages(prev, sessionId, now));
  }, []);

  const setThreadMessages = useCallback((sessionId, threadId, messages) => {
    setSessions((prev) => applySetThreadMessages(prev, sessionId, threadId, messages));
  }, []);

  /**
   * appendMessage supports two modes:
   *  1. Normal: append message to the session's messages array.
   *  2. Replace: if message has __replaceId, replace the message with that id instead of appending.
   *     This is used to swap the loading placeholder with the real assistant response.
   */
  const appendMessage = useCallback((sessionId, threadId, message) => {
    const now = new Date().toISOString();
    setSessions((prev) => applyAppendMessage(prev, sessionId, threadId, message, now));
  }, []);

  const setSessionThreads = useCallback((sessionId, threads) => {
    setSessions((prev) => applySetSessionThreads(prev, sessionId, threads));
  }, []);

  const setActiveThread = useCallback((sessionId, threadId) => {
    setSessions((prev) =>
      sortByLastActive(
        prev.map((session) =>
          session.id === sessionId ? { ...session, active_thread_id: threadId } : session
        )
      )
    );
  }, []);

  const addThread = useCallback((sessionId, thread) => {
    setSessions((prev) =>
      sortByLastActive(
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                threads: [...session.threads, { ...thread, messages: [] }],
                active_thread_id: thread.id,
              }
            : session
        )
      )
    );
  }, []);

  const mergeBackendSessions = useCallback((backendSessions) => {
    setSessions((prev) => {
      const byId = new Map(prev.map((s) => [s.id, s]));
      for (const b of backendSessions) {
        const current = byId.get(b.id);
        byId.set(
          b.id,
          normalizeSessionRecord(b, current, {
            lastActive: current?.last_active || b.updated_at || b.created_at,
          })
        );
      }
      const backendIds = new Set(backendSessions.map((s) => s.id));
      const merged = [...byId.values()].filter((s) => backendIds.has(s.id));
      return sortByLastActive(merged);
    });
  }, []);

  const updateSession = useCallback((sessionId, updates) => {
    setSessions((prev) =>
      sortByLastActive(
        prev.map((s) => (s.id === sessionId ? { ...s, ...updates } : s))
      )
    );
  }, []);

  return {
    sessions,
    addSession,
    deleteSession,
    clearSessionMessages,
    setThreadMessages,
    appendMessage,
    mergeBackendSessions,
    setSessionThreads,
    setActiveThread,
    addThread,
    updateSession,
  };
}
