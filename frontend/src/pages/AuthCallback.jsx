import React from 'react';
import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { exchangeGithubCode } from '../utils/api';

export default function AuthCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState(null);

  useEffect(() => {
    const code = searchParams.get('code');

    if (!code) {
      setError('No authorization code received from GitHub. Please try connecting again.');
      return;
    }

    let cancelled = false;

    exchangeGithubCode(code)
      .then(() => {
        if (cancelled) return;
        navigate('/', { replace: true });
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('[AuthCallback] Exchange failed:', err);
        setError(err.message || 'GitHub connection failed. Please try again.');
      });

    return () => {
      cancelled = true;
    };
  }, []); // Run once on mount

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-base text-center px-6 gap-5">
        <div className="font-mono text-xs text-text-muted uppercase tracking-widest mb-1">Codeseek</div>
        
        <div className="max-w-md border border-border bg-surface-2 p-6 rounded flex flex-col gap-4">
          <p className="text-offline text-sm font-medium">⚠ {error}</p>
        </div>

        <a
          href="/"
          className="text-xs text-text-secondary hover:text-text-primary hover:underline transition-colors font-mono"
        >
          &lt; Back to home
        </a>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-base text-center gap-3">
      <div className="font-mono text-xs text-text-muted uppercase tracking-widest">Codeseek</div>
      <div className="flex items-center gap-2 text-text-secondary text-sm">
        <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
        Connecting to GitHub…
      </div>
    </div>
  );
}
