import React from 'react';
import { useState } from 'react';
import { classifySource } from './sourceCards';

/**
 * Premium source card showing a cited source file, symbol, lines, and category.
 * Clicking the file path copies it to clipboard.
 */
export default function SourceCard({ source }) {
  const [copied, setCopied] = useState(false);
  
  const classified = classifySource(source);
  if (!classified) return null;

  const { file, badge, label, lines, copyValue } = classified;

  const handleCopy = () => {
    if (!file) return;
    navigator.clipboard.writeText(copyValue).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const getBadgeStyle = (b) => {
    switch (b) {
      case 'Docs':
        return 'border-online/25 text-online bg-online/5';
      case 'Test':
        return 'border-warning/25 text-warning bg-warning/5';
      case 'Config':
        return 'border-text-secondary/25 text-text-secondary bg-text-secondary/5';
      case 'Generated report':
        return 'border-offline/25 text-offline bg-offline/5';
      case 'Code':
      default:
        return 'border-accent-dim/25 text-accent-dim bg-accent-dim/5';
    }
  };

  return (
    <div className="inline-flex items-center gap-1.5 bg-surface-3 border border-border/80 rounded-md px-2 py-0.5 text-2xs font-mono select-none">
      {/* Badge Type */}
      <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold tracking-wide border ${getBadgeStyle(badge)}`}>
        {badge}
      </span>

      {/* Copyable Path/Symbol */}
      <button
        onClick={handleCopy}
        title="Copy path and details"
        className="text-text-primary hover:text-text-secondary transition-colors font-medium truncate max-w-[260px] text-[10.5px]"
      >
        {copied ? '✓ copied' : label || 'unknown source'}
      </button>

      {/* Line Ranges */}
      {lines && (
        <span className="text-text-muted shrink-0 text-[9.5px] border-l border-border-subtle/50 pl-1.5">
          lines {lines.replace('-', '–')}
        </span>
      )}
    </div>
  );
}
