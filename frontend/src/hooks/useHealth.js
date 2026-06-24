import { useState, useEffect } from 'react';
import { fetchHealth } from '../utils/api';

const POLL_INTERVAL_MS = 60_000;

/**
 * Polls the backend health endpoint on mount and every 60 seconds.
 * Returns status: "checking" | "online" | "offline"
 */
export function useHealth() {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      const alive = await fetchHealth();
      if (!cancelled) setStatus(alive ? 'online' : 'offline');
    };

    check();
    const interval = setInterval(check, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { status };
}
