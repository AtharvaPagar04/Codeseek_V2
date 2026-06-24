import { useEffect, useRef } from 'react';

/**
 * A simple confirmation dialog modal.
 * Props: message, onConfirm, onCancel, confirmLabel (default "Delete"), danger (default true)
 */
export default function ConfirmDialog({ message, onConfirm, onCancel, confirmLabel = 'Delete', danger = true }) {
  const cancelRef = useRef(null);

  // Close on Escape
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onCancel]);

  // Focus cancel button on mount for keyboard safety
  useEffect(() => {
    cancelRef.current?.focus();
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={(e) => e.target === e.currentTarget && onCancel()}
      aria-modal="true"
      role="alertdialog"
    >
      <div className="bg-surface-2 border border-border rounded-2xl w-80 p-5 shadow-lg animate-fadeIn">
        <p className="text-text-primary text-sm leading-relaxed">{message}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button
            ref={cancelRef}
            onClick={onCancel}
            className="px-3 py-1.5 text-sm text-text-secondary border border-border rounded-lg hover:border-text-muted hover:text-text-primary transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
              danger
                ? 'bg-offline/10 text-offline border border-offline/30 hover:bg-offline/20'
                : 'bg-surface-3 text-text-primary border border-border hover:bg-surface-2'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
