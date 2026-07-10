import React, { useRef, useCallback, useEffect, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { getNeighborIds } from './graphTransform';
import {
  NODE_COLORS,
  NODE_GLOW,
  EDGE_COLORS,
  EDGE_HIGHLIGHT_COLORS,
  EDGE_INBOUND_HIGHLIGHT_COLORS,
  GRAPH_BG,
  DIM_OPACITY,
  LINK_WIDTH_DEFAULT,
  LINK_WIDTH_HIGHLIGHT,
  NODE_BORDER_WIDTH,
  SELECTED_RING_WIDTH,
} from './graphTheme';

// Hex color adjustment utility for radial gradients and borders
function adjustColorBrightness(hex, percent) {
  if (!hex || !hex.startsWith('#')) return hex;
  let color = hex.slice(1);
  if (color.length === 3) {
    color = color.split('').map(c => c + c).join('');
  }
  const num = parseInt(color, 16);
  const amt = Math.round(2.55 * percent);
  let r = (num >> 16) + amt;
  let g = ((num >> 8) & 0x00ff) + amt;
  let b = (num & 0x0000ff) + amt;
  
  r = Math.max(0, Math.min(255, r));
  g = Math.max(0, Math.min(255, g));
  b = Math.max(0, Math.min(255, b));
  
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

// RGBA alpha modifier
function setAlpha(colorStr, alpha) {
  if (!colorStr) return `rgba(255, 255, 255, ${alpha})`;
  if (colorStr.startsWith('rgba')) {
    return colorStr.replace(/[\d.]+\)$/, `${alpha})`);
  }
  if (colorStr.startsWith('#')) {
    let color = colorStr.slice(1);
    if (color.length === 3) {
      color = color.split('').map(c => c + c).join('');
    }
    const r = parseInt(color.slice(0, 2), 16);
    const g = parseInt(color.slice(2, 4), 16);
    const b = parseInt(color.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return colorStr;
}

// Quadratic curve rounded rect fallback
function drawRoundRect(ctx, x, y, w, h, r) {
  if (ctx.roundRect) {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }
}

// Custom 2D collision force to prevent overlaps and group folders
function forceCollide(radiusFunc) {
  let nodes;
  function force(alpha) {
    if (!nodes) return;
    const n = nodes.length;
    for (let i = 0; i < n; i++) {
      const nodeA = nodes[i];
      const rA = radiusFunc(nodeA);
      for (let j = i + 1; j < n; j++) {
        const nodeB = nodes[j];
        const rB = radiusFunc(nodeB);
        const dx = nodeB.x - nodeA.x;
        const dy = nodeB.y - nodeA.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const minDistance = rA + rB + 4.5; // Gap buffer
        if (distance < minDistance) {
          const overlap = minDistance - distance;
          const pushX = (dx / (distance || 1)) * overlap * 0.5 * alpha;
          const pushY = (dy / (distance || 1)) * overlap * 0.5 * alpha;
          
          if (!nodeA.fx) {
            nodeA.x -= pushX;
            nodeA.y -= pushY;
          }
          if (!nodeB.fx) {
            nodeB.x += pushX;
            nodeB.y += pushY;
          }
        }
      }
    }
  }
  force.initialize = (initNodes) => {
    nodes = initNodes;
  };
  return force;
}

// Custom positioning force to keep disconnected components close to the center
function forceGravity(centerX = 0, centerY = 0, strength = 0.015) {
  let nodes;
  function force(alpha) {
    if (!nodes) return;
    const n = nodes.length;
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      if (!node.fx) {
        node.vx += (centerX - node.x) * strength * alpha;
        node.vy += (centerY - node.y) * strength * alpha;
      }
    }
  }
  force.initialize = (initNodes) => {
    nodes = initNodes;
  };
  return force;
}

export default function RepoGraphCanvas({
  graphData,
  selectedNode,
  selectedNodeId,
  hoveredNodeId,
  onNodeClick,
  onNodeSelect,
  onNodeHover,
  onBackgroundClick,
  fitViewRef,
}) {
  const fgRef = useRef(null);
  const effectiveSelectedNodeId = selectedNode?.id || selectedNodeId;

  // Expose fit-to-view via ref callback
  useEffect(() => {
    if (fitViewRef) {
      fitViewRef.current = () => {
        if (fgRef.current && typeof fgRef.current.zoomToFit === 'function') {
          fgRef.current.zoomToFit(300, 40);
        }
      };
    }
  }, [fitViewRef]);

  // Auto-fit on data change
  useEffect(() => {
    const timer = setTimeout(() => {
      if (fgRef.current && typeof fgRef.current.zoomToFit === 'function' && graphData.nodes.length > 0) {
        fgRef.current.zoomToFit(400, 50);
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [graphData.nodes.length]);

  // 1-hop and 2-hop neighbors for focus mode
  const highlightedId = effectiveSelectedNodeId || hoveredNodeId;
  const neighborIds = useMemo(() => {
    if (!highlightedId) return new Set();
    return getNeighborIds(highlightedId, graphData.links);
  }, [highlightedId, graphData.links]);

  const secondNeighborIds = useMemo(() => {
    if (!highlightedId) return new Set();
    const set = new Set();
    for (const neighborId of neighborIds) {
      const neighbors = getNeighborIds(neighborId, graphData.links);
      for (const nid of neighbors) {
        if (nid !== highlightedId && !neighborIds.has(nid)) {
          set.add(nid);
        }
      }
    }
    return set;
  }, [highlightedId, neighborIds, graphData.links]);

  // Adaptive forces configuration
  useEffect(() => {
    if (!fgRef.current || typeof fgRef.current.d3Force !== 'function' || typeof fgRef.current.d3ReheatSimulation !== 'function') return;
    const fg = fgRef.current;
    const numNodes = graphData.nodes.length;
    if (numNodes === 0) return;

    const numLinks = graphData.links.length;
    const avgDegree = numNodes > 0 ? (2 * numLinks) / numNodes : 0;

    // Charge force (repulsion) - balanced so disconnected nodes do not shoot off
    const chargeStrength = -120 - Math.min(numNodes * 1.2, 180);
    const chargeForce = fg.d3Force('charge');
    if (chargeForce) {
      chargeForce.strength(chargeStrength);
    }

    // Link force (distance)
    const linkForce = fg.d3Force('link');
    if (linkForce) {
      linkForce.distance((link) => {
        const type = link.type || 'unknown';
        if (type === 'contains') {
          return 25 + Math.min(numNodes * 0.05, 15);
        }
        if (type === 'imports') {
          return 60 + Math.min(numNodes * 0.1, 40);
        }
        return 40 + Math.min(numNodes * 0.08, 25);
      });
    }

    // Collision force registration
    fg.d3Force('collision', forceCollide((node) => node.size || 5));
    
    // Gravity force registration to keep disconnected components close to main graph
    const gravityStrength = 0.012 + Math.min(numNodes * 0.0001, 0.015);
    fg.d3Force('gravity', forceGravity(0, 0, gravityStrength));

    // Re-heat simulation
    fg.d3ReheatSimulation();
  }, [graphData.nodes.length, graphData.links.length]);

  // Track label bounds on each frame to prevent overlaps
  const frameOccupiedRectsRef = useRef([]);

  // Custom node renderer
  const nodeCanvasObject = useCallback(
    (node, ctx, globalScale) => {
      const isSelected = node.id === effectiveSelectedNodeId;
      const isHovered = node.id === hoveredNodeId;
      const is1stDegree = neighborIds.has(node.id);
      const is2ndDegree = secondNeighborIds.has(node.id);

      // Focus mode opacity target
      let targetOpacity = 1.0;
      if (highlightedId) {
        if (node.id === highlightedId) {
          targetOpacity = 1.0;
        } else if (is1stDegree) {
          targetOpacity = 0.95;
        } else if (is2ndDegree) {
          targetOpacity = 0.45;
        } else {
          targetOpacity = 0.06;
        }
      }

      // Faded look for dropped nodes (visually differentiates them from considered nodes)
      if (node.is_trace_dropped) {
        targetOpacity *= 0.35;
      }

      // Smooth opacity interpolation
      if (node.currentOpacity === undefined) {
        node.currentOpacity = targetOpacity;
      } else {
        node.currentOpacity += (targetOpacity - node.currentOpacity) * 0.18;
      }

      // Scale factor target based on focus/hover/select
      let targetScale = 1.0;
      if (isSelected) {
        targetScale = 1.35;
      } else if (isHovered) {
        targetScale = 1.25;
      } else if (highlightedId) {
        targetScale = is1stDegree ? 1.05 : 0.85;
      }

      // Smooth scale interpolation
      if (node.currentScale === undefined) {
        node.currentScale = targetScale;
      } else {
        node.currentScale += (targetScale - node.currentScale) * 0.18;
      }

      const baseColor = nodeColor(node);

      // Defensive coordinates and sizing calculations
      const nx = Number.isFinite(node.x) ? node.x : 0;
      const ny = Number.isFinite(node.y) ? node.y : 0;
      const rawSize = (node.size || 4) * (node.currentScale || 1.0);
      const size = Number.isFinite(rawSize) && rawSize > 0 ? rawSize : 4;

      ctx.save();
      ctx.globalAlpha = Number.isFinite(node.currentOpacity) ? node.currentOpacity : 1.0;

      // 1. Premium radial gradient node
      let fillStyle = baseColor;
      if (Number.isFinite(nx) && Number.isFinite(ny) && Number.isFinite(size) && size > 0) {
        try {
          const grad = ctx.createRadialGradient(
            nx - size * 0.12,
            ny - size * 0.12,
            size * 0.05,
            nx,
            ny,
            size
          );
          grad.addColorStop(0, adjustColorBrightness(baseColor, 55)); // 3D center highlight preserving hue
          grad.addColorStop(0.2, adjustColorBrightness(baseColor, 15));
          grad.addColorStop(0.8, baseColor);
          grad.addColorStop(1, adjustColorBrightness(baseColor, -25)); // Depth edge
          fillStyle = grad;
        } catch (e) {
          fillStyle = baseColor;
        }
      }

      // Glow for selected, hovered, or active retrieval states
      const hasGlow = !highlightedId || node.id === highlightedId || is1stDegree;
      if (hasGlow && (isSelected || isHovered || node.is_retrieved || node.is_graph_active || node.is_trace_context || node.is_trace_final || node.is_trace_cited || node.is_trace_dropped)) {
        let glowColor = baseColor;
        if (node.is_trace_cited) glowColor = NODE_GLOW.cited;
        else if (node.is_trace_final) glowColor = NODE_GLOW.final;
        else if (node.is_trace_context) glowColor = NODE_GLOW.context_selected || '#2dd4bf';
        else if (node.is_graph_active) glowColor = NODE_GLOW.graph_active;
        else if (node.is_trace_dropped) glowColor = NODE_GLOW.dropped;
        else if (node.is_retrieved) glowColor = NODE_GLOW.retrieved;
        else if (isSelected) glowColor = NODE_GLOW.selected || '#ffffff';

        ctx.shadowColor = glowColor;
        const glowIntensity = isSelected ? 16 : isHovered ? 12 : 8;
        ctx.shadowBlur = glowIntensity * (Number.isFinite(node.currentScale) ? node.currentScale : 1.0);
      }

      // Base circle
      ctx.beginPath();
      ctx.arc(nx, ny, size, 0, 2 * Math.PI);
      ctx.fillStyle = fillStyle;
      ctx.fill();

      ctx.shadowBlur = 0; // Clear shadow for borders

      // Stroke outline & selection indicator
      if (isSelected) {
        // Inner white border
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.0 / globalScale;
        ctx.stroke();

        // Outer semi-transparent selection ring for premium visualization feel
        ctx.beginPath();
        ctx.arc(nx, ny, size + 3.0 / globalScale, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.lineWidth = 1.0 / globalScale;
        ctx.stroke();
      } else if (isHovered) {
        ctx.strokeStyle = '#f1f5f9';
        ctx.lineWidth = 1.4 / globalScale;
        ctx.stroke();
      } else {
        ctx.strokeStyle = adjustColorBrightness(baseColor, -40);
        ctx.lineWidth = 0.8 / globalScale;
        ctx.stroke();
      }

      // 2. Rounded background labels with collision prevention
      const showLabel =
        (highlightedId && (node.id === highlightedId || is1stDegree)) ||
        isSelected ||
        isHovered ||
        (node.importance > 0.7 && globalScale > 0.7) ||
        globalScale > 1.6;

      if (showLabel && (Number.isFinite(node.currentOpacity) ? node.currentOpacity : 0) > 0.15) {
        const label = node.label || '';
        const fontSize = Math.max(Math.min(9.5 / globalScale, 18), 3.5);
        ctx.font = `500 ${fontSize}px 'Inter', sans-serif`;
        
        const textWidth = ctx.measureText(label).width;
        const paddingX = 4;
        const paddingY = 2;
        const rectW = textWidth + paddingX * 2;
        const rectH = fontSize + paddingY * 2;
        const rectX = nx - rectW / 2;
        const rectY = ny + size + 2;

        const box = {
          x1: rectX,
          y1: rectY,
          x2: rectX + rectW,
          y2: rectY + rectH
        };

        // Reset frames tracking list on processing the first node
        const isFirstNode = node.id === graphData.nodes[0]?.id;
        if (isFirstNode) {
          frameOccupiedRectsRef.current = [];
        }

        const overlaps = frameOccupiedRectsRef.current.some(other => {
          return !(box.x2 < other.x1 || box.x1 > other.x2 || box.y2 < other.y1 || box.y1 > other.y2);
        });

        if (!overlaps || isSelected || isHovered) {
          if (!overlaps) {
            frameOccupiedRectsRef.current.push(box);
          }

          ctx.beginPath();
          drawRoundRect(ctx, rectX, rectY, rectW, rectH, 3);
          ctx.fillStyle = 'rgba(6, 8, 16, 0.75)';
          ctx.fill();
          ctx.strokeStyle = isSelected || isHovered ? 'rgba(255, 255, 255, 0.18)' : 'rgba(255, 255, 255, 0.05)';
          ctx.lineWidth = 0.5 / globalScale;
          ctx.stroke();

          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillStyle = isSelected || isHovered ? '#ffffff' : (highlightedId && is1stDegree ? '#e2e8f0' : '#94a3b8');
          ctx.fillText(label, nx, rectY + paddingY);
        }
      }

      ctx.restore();
    },
    [effectiveSelectedNodeId, hoveredNodeId, highlightedId, neighborIds, secondNeighborIds, graphData.nodes]
  );

  // Link styling with focus mode opacities and type highlights
  const linkColor = useCallback(
    (link) => {
      const type = link.type || 'unknown';
      let baseColor = EDGE_COLORS[type] || EDGE_COLORS.unknown;

      const src = typeof link.source === 'object' ? link.source.id : link.source;
      const tgt = typeof link.target === 'object' ? link.target.id : link.target;

      if (highlightedId) {
        const isOutbound = src === highlightedId;
        const isInbound = tgt === highlightedId;

        if (isOutbound) {
          baseColor = type === 'imports'
            ? EDGE_INBOUND_HIGHLIGHT_COLORS.imports
            : (EDGE_HIGHLIGHT_COLORS[type] || EDGE_HIGHLIGHT_COLORS.unknown);
        } else if (isInbound) {
          baseColor = type === 'imports'
            ? EDGE_HIGHLIGHT_COLORS.imports
            : (EDGE_INBOUND_HIGHLIGHT_COLORS[type] || EDGE_INBOUND_HIGHLIGHT_COLORS.unknown);
        }
      }

      // Calculate dynamic opacity for links
      let opacity = 1.0;
      if (highlightedId) {
        const isDirect = src === highlightedId || tgt === highlightedId;
        const isSecondary = (neighborIds.has(src) && secondNeighborIds.has(tgt)) ||
                            (neighborIds.has(tgt) && secondNeighborIds.has(src)) ||
                            (neighborIds.has(src) && neighborIds.has(tgt));
        if (isDirect) {
          opacity = 0.85;
        } else if (isSecondary) {
          opacity = 0.35;
        } else {
          opacity = 0.04; // Fade unrelated links heavily
        }
        return setAlpha(baseColor, opacity);
      }
      
      return baseColor;
    },
    [highlightedId, neighborIds, secondNeighborIds]
  );

  const linkWidth = useCallback(
    (link) => {
      const weight = link.weight ? Math.max(LINK_WIDTH_DEFAULT, Math.min(Number(link.weight), 3)) : LINK_WIDTH_DEFAULT;
      
      if (!highlightedId) {
        return weight;
      }
      
      const src = typeof link.source === 'object' ? link.source.id : link.source;
      const tgt = typeof link.target === 'object' ? link.target.id : link.target;
      
      const isDirect = src === highlightedId || tgt === highlightedId;
      if (isDirect) {
        return Math.max(LINK_WIDTH_HIGHLIGHT, weight * 1.5);
      }
      
      const isSecondary = (neighborIds.has(src) && secondNeighborIds.has(tgt)) ||
                          (neighborIds.has(tgt) && secondNeighborIds.has(src)) ||
                          (neighborIds.has(src) && neighborIds.has(tgt));
      if (isSecondary) {
        return weight * 1.1;
      }
      
      return weight * 0.7;
    },
    [highlightedId, neighborIds, secondNeighborIds]
  );

  const handleNodeClick = useCallback(
    (node) => {
      onNodeClick?.(node.id);
      onNodeSelect?.(node);
    },
    [onNodeClick, onNodeSelect]
  );

  const handleNodeHover = useCallback(
    (node) => {
      onNodeHover(node ? node.id : null);
    },
    [onNodeHover]
  );

  const handleBackgroundClick = useCallback(() => {
    onBackgroundClick();
  }, [onBackgroundClick]);

  // Handle Escape key
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onBackgroundClick();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onBackgroundClick]);

  if (graphData.nodes.length === 0) {
    const emptyMessage = graphData.summary?.message || 'Adjust filters or change graph mode.';
    return (
      <div className="flex-1 flex items-center justify-center" style={{ backgroundColor: GRAPH_BG }}>
        <div className="text-center space-y-2 max-w-xs">
          <svg className="w-10 h-10 mx-auto text-text-muted/20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <p className="text-xs text-text-muted font-mono">No graph data to display</p>
          <p className="text-[10px] text-text-muted/60">{emptyMessage}</p>
        </div>
      </div>
    );
  }

  // Calculate dynamic physics boundaries
  const numNodes = graphData.nodes.length;
  const d3AlphaDecay = Math.max(0.02, Math.min(0.06, 12 / (numNodes + 200)));
  const d3VelocityDecay = Math.max(0.2, Math.min(0.5, 0.15 + numNodes * 0.001));
  const cooldownTicks = Math.max(80, Math.min(180, 50 + Math.round(numNodes * 0.4)));

  return (
    <div className="relative min-w-0 flex-1" style={{ backgroundColor: GRAPH_BG }}>
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        backgroundColor={GRAPH_BG}
        nodeCanvasObject={nodeCanvasObject}
        nodePointerAreaPaint={(node, color, ctx) => {
          const nx = Number.isFinite(node.x) ? node.x : 0;
          const ny = Number.isFinite(node.y) ? node.y : 0;
          const rawSize = (node.size || 4) * (node.currentScale || 1.0);
          const size = Number.isFinite(rawSize) && rawSize > 0 ? rawSize : 4;
          ctx.beginPath();
          ctx.arc(nx, ny, size + 2, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={(link) => {
          const src = typeof link.source === 'object' ? link.source.id : link.source;
          const tgt = typeof link.target === 'object' ? link.target.id : link.target;
          const isDirect = highlightedId && (src === highlightedId || tgt === highlightedId);
          return isDirect ? 4.5 : 3.0;
        }}
        linkDirectionalArrowRelPos={0.9}
        linkDirectionalArrowColor={linkColor}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onBackgroundClick={handleBackgroundClick}
        onNodeDragEnd={(node) => {
          if (node.type === 'folder') {
            node.fx = node.x;
            node.fy = node.y;
          } else {
            node.fx = null;
            node.fy = null;
          }
          if (fgRef.current && typeof fgRef.current.d3ReheatSimulation === 'function') {
            fgRef.current.d3ReheatSimulation();
          }
        }}
        cooldownTicks={cooldownTicks}
        d3AlphaDecay={d3AlphaDecay}
        d3VelocityDecay={d3VelocityDecay}
        enableNodeDrag={true}
        enableZoomPanInteraction={true}
        minZoom={0.2}
        maxZoom={8}
      />

      {/* Overlay stats */}
      <div className="absolute bottom-3 left-3 bg-surface-2/80 backdrop-blur-sm border border-border rounded-lg px-2.5 py-1.5 text-[9px] font-mono text-text-muted pointer-events-none select-none">
        {graphData.nodes.length} nodes · {graphData.links.length} edges
      </div>
    </div>
  );
}

function nodeColor(node) {
  if (node?.type === 'chunk') {
    if (node.is_trace_cited) return '#f8fafc';
    if (node.is_trace_final) return '#60a5fa';
    if (node.is_trace_context) return '#2dd4bf';
    if (node.is_trace_dropped) return '#ef4444';
    if (node.is_graph_active) return '#4ade80';
    if (node.is_retrieved) return '#64748b';
  }
  return NODE_COLORS[node.type] || NODE_COLORS.unknown;
}

