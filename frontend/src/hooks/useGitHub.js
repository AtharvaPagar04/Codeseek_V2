import { useState, useCallback, useEffect } from 'react';
import {
  connectGithubToken,
  fetchGithubSessionMe,
  listGithubRepos,
  logoutGithubSession,
} from '../utils/api';

const API_BASE = import.meta.env?.VITE_API_BASE_URL?.replace(/\/$/, "") || 'http://127.0.0.1:8000';

export function useGitHub() {
  const [username, setUsername] = useState(null);
  const [avatarUrl, setAvatarUrl] = useState(null);
  const [repos, setRepos] = useState([]);
  const [reposLoading, setReposLoading] = useState(false);
  const [reposError, setReposError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [oauthLoading, setOauthLoading] = useState(false);
  const [oauthError, setOauthError] = useState(null);
  const [authStateMessage, setAuthStateMessage] = useState(null);

  const loadAuthState = useCallback(async () => {
    try {
      const data = await fetchGithubSessionMe();
      if (!data?.authenticated || !data.user) {
        setAuthStateMessage((current) =>
          isConnected ? 'GitHub session expired. Reconnect to list repositories and create sessions.' : current
        );
        setIsConnected(false);
        setUsername(null);
        setAvatarUrl(null);
        return;
      }
      setAuthStateMessage(null);
      setIsConnected(true);
      setUsername(data.user.username || null);
      setAvatarUrl(data.user.avatar_url || null);
    } catch {
      setAuthStateMessage((current) =>
        isConnected ? 'GitHub session expired. Reconnect to continue.' : current
      );
      setIsConnected(false);
    }
  }, [isConnected]);

  useEffect(() => {
    loadAuthState();
  }, [loadAuthState]);

  const initiateOAuth = useCallback(() => {
    setOauthError(null);
    setOauthLoading(true);

    const loginUrl = `${API_BASE}/auth/github/login`;
    const popup = window.open(
      loginUrl,
      'codeseek_github_login',
      'width=600,height=720,scrollbars=yes,resizable=yes,left=200,top=100'
    );

    // Popup blocked — fall back to full-page redirect
    if (!popup || popup.closed) {
      setOauthLoading(false);
      window.location.href = loginUrl;
      return;
    }

    const handleMessage = (event) => {
      // Only accept from our backend origin (where the popup page is served)
      if (
        event.data?.type !== 'CODESEEK_GITHUB_AUTH' ||
        !event.origin.includes(new URL(API_BASE).hostname)
      ) return;

      cleanup();

      if (event.data.status === 'success') {
        loadAuthState().finally(() => setOauthLoading(false));
      } else {
        setOauthError(event.data.error || 'GitHub login failed. Please try again.');
        setOauthLoading(false);
      }
    };

    const pollTimer = setInterval(() => {
      if (popup.closed) {
        cleanup();
        // Popup closed without postMessage (user closed it) — try refresh anyway
        loadAuthState().finally(() => setOauthLoading(false));
      }
    }, 500);

    const cleanup = () => {
      clearInterval(pollTimer);
      window.removeEventListener('message', handleMessage);
    };

    window.addEventListener('message', handleMessage);
  }, [loadAuthState]);

  const storeAuth = useCallback(async (accessToken) => {
    const data = await connectGithubToken(accessToken);
    setIsConnected(true);
    setUsername(data.username || null);
    setAvatarUrl(data.avatar_url || null);
  }, []);

  const fetchRepos = useCallback(async () => {
    if (!isConnected) return;
    setReposLoading(true);
    setReposError(null);
    try {
      const data = await listGithubRepos();
      setRepos(data);
    } catch (err) {
      setReposError(err.message || 'Could not load repositories. Check your GitHub connection.');
    } finally {
      setReposLoading(false);
    }
  }, [isConnected]);

  const disconnect = useCallback(async () => {
    try {
      await logoutGithubSession();
    } catch (err) {
      console.warn('[useGitHub] backend logout failed:', err?.message || err);
    }
    setIsConnected(false);
    setUsername(null);
    setAvatarUrl(null);
    setRepos([]);
    setAuthStateMessage(null);
  }, []);

  return {
    isConnected,
    username,
    avatarUrl,
    repos,
    reposLoading,
    reposError,
    oauthLoading,
    oauthError,
    authStateMessage,
    initiateOAuth,
    storeAuth,
    fetchRepos,
    disconnect,
  };
}
