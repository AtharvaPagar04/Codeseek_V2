/**
 * Graph visualization theme — colors, sizes, and visual constants.
 * Aligned with CodeSeek's dark monochrome design system.
 */

export const NODE_COLORS = {
  folder:   '#38bdf8', // sky-400
  file:     '#a78bfa', // violet-400
  symbol:   '#f472b6', // pink-400
  external: '#fb923c', // orange-400
  query:    '#22d3ee', // cyan-400
  chunk:    '#64748b', // slate-500
  answer:   '#34d399', // emerald-400
  unknown:  '#737373', // neutral-500
};

export const NODE_GLOW = {
  retrieved:    '#22d3ee', // cyan-400
  graph_active: '#4ade80', // green-400
  final:        '#60a5fa', // blue-400
  context_selected: '#2dd4bf', // teal-400
  cited:        '#f8fafc', // slate-50
  dropped:      '#fb7185', // rose-400
  selected:     '#ffffff',
  hovered:      '#e2e8f0',
};

export const EDGE_COLORS = {
  contains:  'rgba(100, 116, 139, 0.15)', // slate muted
  imports:   'rgba(239, 68, 68, 0.20)',   // red
  defines:   'rgba(244, 114, 182, 0.20)', // pink
  retrieval: 'rgba(34, 211, 238, 0.25)',  // cyan
  retrieved: 'rgba(148, 163, 184, 0.20)',
  graph_added: 'rgba(168, 85, 247, 0.35)',
  context_selected: 'rgba(45, 212, 191, 0.35)',
  final_context: 'rgba(96, 165, 250, 0.35)',
  cited: 'rgba(52, 211, 153, 0.40)',
  dropped: 'rgba(251, 113, 133, 0.18)',
  unknown:   'rgba(115, 115, 115, 0.12)',
};

export const EDGE_HIGHLIGHT_COLORS = {
  contains:  'rgba(148, 163, 184, 0.70)',
  imports:   'rgba(239, 68, 68, 0.85)',   // bright red
  defines:   'rgba(244, 114, 182, 0.80)',
  retrieval: 'rgba(34, 211, 238, 0.90)',
  retrieved: 'rgba(203, 213, 225, 0.90)',
  graph_added: 'rgba(192, 132, 252, 0.95)',
  context_selected: 'rgba(94, 234, 212, 0.95)',
  final_context: 'rgba(147, 197, 253, 0.95)',
  cited: 'rgba(110, 231, 183, 1)',
  dropped: 'rgba(251, 113, 133, 0.75)',
  unknown:   'rgba(163, 163, 163, 0.60)',
};

export const EDGE_INBOUND_HIGHLIGHT_COLORS = {
  contains:  'rgba(56, 189, 248, 0.85)',  // Sky blue (contained in parent)
  imports:   'rgba(252, 165, 165, 0.90)',  // Light red (imported by parent)
  defines:   'rgba(34, 211, 238, 0.85)',  // Cyan (defined in file)
  retrieved: 'rgba(203, 213, 225, 0.90)',
  graph_added: 'rgba(192, 132, 252, 0.95)',
  context_selected: 'rgba(94, 234, 212, 0.95)',
  final_context: 'rgba(147, 197, 253, 0.95)',
  cited: 'rgba(110, 231, 183, 1)',
  dropped: 'rgba(251, 113, 133, 0.75)',
  unknown:   'rgba(244, 63, 94, 0.80)',   // Rose for other inbound links
};

export const NODE_SIZES = {
  folder:   7,
  file:     5,
  symbol:   4,
  external: 4,
  query:    8,
  chunk:    5,
  answer:   8,
  unknown:  3,
};

export const GRAPH_BG = '#060810';

export const LABEL_COLORS = {
  default: '#d4d4d4',
  dimmed:  '#555555',
};

export const DIM_OPACITY = 0.12;
export const LINK_WIDTH_DEFAULT = 0.6;
export const LINK_WIDTH_HIGHLIGHT = 1.8;
export const NODE_BORDER_WIDTH = 1.5;
export const SELECTED_RING_WIDTH = 3;
