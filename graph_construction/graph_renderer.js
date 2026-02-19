// Graph rendering with Dagre layout and normalized coordinates
// Updated to use edge thickness for thought length instead of node size

// ==================== Layout and Coordinate Normalization ====================
function layoutGraph() {
    // Create dagre graph
    const g = new dagre.graphlib.Graph({ multigraph: true });
    g.setGraph({
        rankdir: 'LR',
        ranksep: 120,
        nodesep: 60,
        edgesep: 40,
        marginx: 40,
        marginy: 40
    });
    g.setDefaultEdgeLabel(() => ({}));
    
    // Add nodes with sizing based on label content.
    // Apply verbosity setting to choose which label to use for sizing.
    nodesData.forEach(node => {
        const label = settings.nodeVerbosity ? node.label_verbose : node.label_minimal;
        node.displayLabel = label;  // Store for rendering
        
        const lines = label.split('\\n');
        const line1Len = lines[0] ? Math.min(lines[0].length, 30) : 0;
        const restMax  = lines.slice(1).reduce((m, l) => Math.max(m, Math.min(l.length, 35)), 0);
        const widthFromL1   = line1Len  * 7.5 + 24;
        const widthFromRest = restMax   * 5.5 + 24;
        const width  = Math.max(100, widthFromL1, widthFromRest);
        const height = Math.max(40, lines.length * 16 + 12);
        g.setNode(node.id, { width, height, ...node });
    });
    
    // Add edges - use unique names for multigraph support
    edgesData.forEach((edge, idx) => {
        g.setEdge(edge.from, edge.to, edge, `edge-${idx}`);
    });
    
    // Layout
    dagre.layout(g);
    
    // Calculate bounding box
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    g.nodes().forEach(nodeId => {
        const node = g.node(nodeId);
        const left = node.x - node.width / 2;
        const right = node.x + node.width / 2;
        const top = node.y - node.height / 2;
        const bottom = node.y + node.height / 2;
        
        minX = Math.min(minX, left);
        maxX = Math.max(maxX, right);
        minY = Math.min(minY, top);
        maxY = Math.max(maxY, bottom);
    });
    
    // Add padding
    const padding = 40;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;
    
    // Normalize coordinates to start at (0, 0)
    const offsetX = -minX;
    const offsetY = -minY;
    
    // Update node positions
    g.nodes().forEach(nodeId => {
        const node = g.node(nodeId);
        node.x += offsetX;
        node.y += offsetY;
    });
    
    // Update edge points
    g.edges().forEach(e => {
        const edge = g.edge(e);
        if (edge.points) {
            edge.points.forEach(point => {
                point.x += offsetX;
                point.y += offsetY;
            });
        }
    });
    
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    
    return { g, graphWidth, graphHeight };
}

// ==================== SVG Creation ====================
function createSVG(graphWidth, graphHeight) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${graphWidth} ${graphHeight}`);
    svg.setAttribute('width', graphWidth);
    svg.setAttribute('height', graphHeight);
    svg.style.overflow = 'visible';
    return svg;
}

// ==================== Markers ====================
function createMarkers(svg) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    
    // Exec arrow (regular)
    const markerExec = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerExec.setAttribute('id', 'arrowhead-exec');
    markerExec.setAttribute('markerWidth', '10');
    markerExec.setAttribute('markerHeight', '10');
    markerExec.setAttribute('refX', '9');
    markerExec.setAttribute('refY', '3');
    markerExec.setAttribute('orient', 'auto');
    const pathExec = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathExec.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathExec.setAttribute('class', 'arrowhead');
    markerExec.appendChild(pathExec);
    defs.appendChild(markerExec);
    
    // Exec arrow for multi-node steps (blue)
    const markerExecMulti = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerExecMulti.setAttribute('id', 'arrowhead-exec-multi');
    markerExecMulti.setAttribute('markerWidth', '10');
    markerExecMulti.setAttribute('markerHeight', '10');
    markerExecMulti.setAttribute('refX', '9');
    markerExecMulti.setAttribute('refY', '3');
    markerExecMulti.setAttribute('orient', 'auto');
    const pathExecMulti = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathExecMulti.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathExecMulti.setAttribute('fill', '#3498db');
    markerExecMulti.appendChild(pathExecMulti);
    defs.appendChild(markerExecMulti);
    
    // Hier arrow
    const markerHier = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerHier.setAttribute('id', 'arrowhead-hier');
    markerHier.setAttribute('markerWidth', '10');
    markerHier.setAttribute('markerHeight', '10');
    markerHier.setAttribute('refX', '9');
    markerHier.setAttribute('refY', '3');
    markerHier.setAttribute('orient', 'auto');
    const pathHier = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathHier.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathHier.setAttribute('class', 'arrowhead hier');
    markerHier.appendChild(pathHier);
    defs.appendChild(markerHier);
    
    svg.appendChild(defs);
    return defs;
}

// ==================== Edge Rendering ====================

/**
 * Map a thought_length to a stroke width.
 * Uses settings.thoughtQuotes to choose raw or clean length.
 */
function getThoughtLength(edge) {
    return settings.thoughtQuotes ? edge.thought_length_clean : edge.thought_length_raw;
}

function thoughtToWidth(thoughtLength) {
    if (thoughtLength <= 0) return 1;
    const capped = Math.min(thoughtLength, 5000);
    if (capped <= 200)  return 6 + (capped / 200) * 6;
    if (capped <= 800)  return 12   + ((capped - 200) / 600) * 12;
    return 24 + ((capped - 800) / 700) * 12;
}

function calculateEdgeStyle(edge) {
    if (edge.type === 'hier') {
        return {
            strokeWidth:    1.5,
            strokeDasharray: '6,4',
            stroke:          '#27ae60',
            markerEnd:       'url(#arrowhead-hier)',
            opacity:         0.75,
        };
    }

    if (edge.type === 'exec') {
        if (edge.is_multi_node_step) {
            return {
                strokeWidth:    1,
                strokeDasharray: '4,4',
                stroke:          '#3498db',
                markerEnd:       'url(#arrowhead-exec-multi)',
                opacity:         0.9,
            };
        }
        const tlen = getThoughtLength(edge);
        if (tlen === 0) {
            return {
                strokeWidth:    1,
                strokeDasharray: '4,4',
                stroke:          '#95a5a6',
                markerEnd:       'url(#arrowhead-exec)',
                opacity:         0.75,
            };
        }
        const w = thoughtToWidth(tlen);
        return {
            strokeWidth:    w,
            strokeDasharray: '',
            stroke:          '#7f8c8d',
            markerEnd:       `url(#arrowhead-exec-w${Math.round(w)})`,
            opacity:         1,
        };
    }

    return { strokeWidth: 1, strokeDasharray: '', stroke: '#bbb',
             markerEnd: 'url(#arrowhead-exec)', opacity: 1 };
}

/**
 * Build a smooth cubic-bezier path string from dagre waypoints.
 * Dagre returns 3+ collinear-ish points; we turn them into a smooth spline.
 */
function pointsToPath(points, offsetY) {
    if (!points || points.length === 0) return '';
    if (points.length === 1) {
        return `M ${points[0].x} ${points[0].y + offsetY}`;
    }
    // Move to first point
    let d = `M ${points[0].x} ${points[0].y + offsetY}`;
    if (points.length === 2) {
        d += ` L ${points[1].x} ${points[1].y + offsetY}`;
        return d;
    }
    // For 3+ points use cubic bezier with control points at 1/3 & 2/3 between segments
    for (let i = 1; i < points.length - 1; i++) {
        const x0 = points[i - 1].x, y0 = points[i - 1].y + offsetY;
        const x1 = points[i].x,     y1 = points[i].y + offsetY;
        const x2 = points[i + 1].x, y2 = points[i + 1].y + offsetY;
        const cpx1 = x0 + (x1 - x0) * 0.67;
        const cpy1 = y0 + (y1 - y0) * 0.67;
        const cpx2 = x1 - (x2 - x1) * 0.33;
        const cpy2 = y1 - (y2 - y1) * 0.33;
        d += ` C ${cpx1} ${cpy1} ${cpx2} ${cpy2} ${x1} ${y1}`;
    }
    // Final segment to last point
    const last = points[points.length - 1];
    d += ` L ${last.x} ${last.y + offsetY}`;
    return d;
}

function renderEdges(svg, g, defs) {
    // Pre-compute edge counts per (from,to) pair for multi-edge offsetting
    const edgesByPair = {};
    edgesData.forEach((edge, idx) => {
        const key = `${edge.from}-${edge.to}`;
        if (!edgesByPair[key]) edgesByPair[key] = [];
        edgesByPair[key].push({ ...edge, idx });
    });

    // Create per-width arrowhead markers (so marker scales with line thickness)
    const widthsSeen = new Set();
    edgesData.forEach(edge => {
        if (edge.type === 'exec' && !edge.is_multi_node_step) {
            const tlen = getThoughtLength(edge);
            if (tlen > 0) {
                widthsSeen.add(Math.round(thoughtToWidth(tlen)));
            }
        }
    });
    widthsSeen.forEach(w => {
        const m = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        m.setAttribute('id', `arrowhead-exec-w${w}`);
        m.setAttribute('markerWidth',  String(6 + w));
        m.setAttribute('markerHeight', String(6 + w));
        m.setAttribute('refX', String(5 + w));
        m.setAttribute('refY', String((4 + w) / 2));
        m.setAttribute('orient', 'auto');
        const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        p.setAttribute('d', `M0,0 L0,${4 + w} L${5 + w},${(4 + w) / 2} z`);
        p.setAttribute('fill', '#7f8c8d');
        m.appendChild(p);
        defs.appendChild(m);
    });

    g.edges().forEach(e => {
        const edge      = g.edge(e);
        const edgeKey   = `${e.v}-${e.w}`;
        const edgesInPair = edgesByPair[edgeKey] || [];
        const edgeIndex = edgesInPair.findIndex(
            ed => ed.type === edge.type && ed.label === edge.label
        );
        const totalEdges = edgesInPair.length;

        const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        edgeGroup.setAttribute('class', `edge ${edge.type}`);

        const style  = calculateEdgeStyle(edge);
        const points = edge.points;

        let offsetY = 0;
        if (totalEdges > 1) {
            offsetY = (edgeIndex - (totalEdges - 1) / 2) * 14;
        }

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d',            pointsToPath(points, offsetY));
        path.setAttribute('stroke',       style.stroke);
        path.setAttribute('stroke-width', style.strokeWidth);
        path.setAttribute('fill',         'none');
        path.setAttribute('opacity',      style.opacity);
        if (style.strokeDasharray) {
            path.setAttribute('stroke-dasharray', style.strokeDasharray);
        }
        path.setAttribute('marker-end', style.markerEnd);

        edgeGroup.appendChild(path);

        // Step-number label (only on exec edges)
        if (edge.label && edge.type === 'exec') {
            const midIdx   = Math.floor(points.length / 2);
            const midPoint = points[midIdx];
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', midPoint.x);
            text.setAttribute('y', midPoint.y + offsetY - 5);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size',   '10');
            text.setAttribute('fill',        '#7f8c8d');
            text.textContent = edge.label;
            edgeGroup.appendChild(text);
        }

        svg.appendChild(edgeGroup);
    });
}

// ==================== Node Rendering ====================
function makeNodeRect(node, fillAttr) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x',      node.x - node.width  / 2);
    rect.setAttribute('y',      node.y - node.height / 2);
    rect.setAttribute('width',  node.width);
    rect.setAttribute('height', node.height);
    rect.setAttribute('rx',     '5');
    rect.setAttribute('ry',     '5');
    rect.setAttribute('fill',   fillAttr);
    // Always set explicit stroke so SVG overrides CSS default
    if (node.has_failure) {
        rect.setAttribute('stroke',       '#e74c3c');
        rect.setAttribute('stroke-width', '3');
    } else {
        rect.setAttribute('stroke',       '#2c3e50');
        rect.setAttribute('stroke-width', '1.5');
    }
    return rect;
}

function renderNodes(svg, g, defs) {
    g.nodes().forEach(nodeId => {
        const node      = g.node(nodeId);
        const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodeGroup.setAttribute('class',        'node');
        nodeGroup.setAttribute('data-id',      nodeId);
        nodeGroup.setAttribute('data-tooltip', node.tooltip);

        if (node.colors && node.colors.length > 1) {
            const gradId = `grad-${nodeId.replace(/[^a-zA-Z0-9]/g, '_')}`;
            const grad   = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            grad.setAttribute('id', gradId);
            grad.setAttribute('x1', '0%'); grad.setAttribute('y1', '0%');
            grad.setAttribute('x2', '100%'); grad.setAttribute('y2', '0%');
            node.colors.forEach((color, i) => {
                const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                s1.setAttribute('offset',     `${i / node.colors.length * 100}%`);
                s1.setAttribute('stop-color', color);
                grad.appendChild(s1);
                const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                s2.setAttribute('offset',     `${(i + 1) / node.colors.length * 100}%`);
                s2.setAttribute('stop-color', color);
                grad.appendChild(s2);
            });
            defs.appendChild(grad);
            nodeGroup.appendChild(makeNodeRect(node, `url(#${gradId})`));
        } else {
            nodeGroup.appendChild(makeNodeRect(node, node.color || '#CFE0F6'));
        }
        
        // Add triangular "hat" for nodes that had cd command stripped
        if (node.has_cd) {
            const hatSize = 12;
            const topY = node.y - node.height / 2;
            const centerX = node.x;
            
            const triangle = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            const d = `M ${centerX} ${topY - hatSize} ` +
                     `L ${centerX - hatSize} ${topY} ` +
                     `L ${centerX + hatSize} ${topY} Z`;
            triangle.setAttribute('d', d);
            triangle.setAttribute('fill', '#f39c12');
            triangle.setAttribute('stroke', '#e67e22');
            triangle.setAttribute('stroke-width', '1.5');
            
            nodeGroup.appendChild(triangle);
        }

        // Add observation indicator rectangle (if enabled and node has observation)
        if (settings.showObservation && node.observation_length > 0) {
            const obsWidth = Math.min(Math.max(node.observation_length / 50, 5), 50);
            const obsHeight = node.height * 0.7;
            const rightX = node.x + node.width / 2;
            const centerY = node.y;

            const obsRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            obsRect.setAttribute('x',      rightX + 4);
            obsRect.setAttribute('y',      centerY - obsHeight / 2);
            obsRect.setAttribute('width',  obsWidth);
            obsRect.setAttribute('height', obsHeight);
            obsRect.setAttribute('rx',     '2');
            obsRect.setAttribute('ry',     '2');

            // Color based on outcome
            let obsFill;
            if (node.observation_outcome === 'success') {
                obsFill = '#7defa7';
            } else if (node.observation_outcome === 'failure') {
                obsFill = '#ff8080';
            } else {
                obsFill = '#bdc3c7';
            }
            obsRect.setAttribute('fill',    obsFill);
            obsRect.setAttribute('opacity', '0.85');
            obsRect.setAttribute('stroke', '#2c3e50');
            obsRect.setAttribute('stroke-width', '0.5');

            nodeGroup.appendChild(obsRect);
        }
        
        const lines = node.displayLabel.split('\\n');
        const lineHeight = 16;
        const totalTextHeight = lines.length * lineHeight;
        const startY = node.y - totalTextHeight / 2 + lineHeight / 2;
        
        lines.forEach((line, i) => {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', node.x);
            text.setAttribute('y', startY + i * lineHeight);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            
            if (i === 0) {
                // Line 1: action title — bold, dark, readable
                text.setAttribute('font-weight', 'bold');
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#1a1a2e');
            } else if (i === 1) {
                // Line 2: step index — medium, slightly muted
                text.setAttribute('font-size', '10');
                text.setAttribute('fill', '#555');
            } else {
                // Lines 3+: path / view-range — small, light grey
                text.setAttribute('font-size', '9');
                text.setAttribute('fill', '#666');
            }
            
            text.textContent = line;
            nodeGroup.appendChild(text);
        });
        
        svg.appendChild(nodeGroup);
    });
}

// ==================== Tooltip ====================
function setupTooltips() {
    const tooltip = document.getElementById('tooltip');
    document.querySelectorAll('.node').forEach(node => {
        node.addEventListener('mouseenter', (e) => {
            const tooltipContent = e.currentTarget.getAttribute('data-tooltip');
            tooltip.innerHTML = tooltipContent;
            tooltip.style.display = 'block';
        });
        
        node.addEventListener('mousemove', (e) => {
            tooltip.style.left = (e.pageX + 15) + 'px';
            tooltip.style.top = (e.pageY + 15) + 'px';
        });
        
        node.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
    });
}

// ==================== Zoom and Pan Controls ====================
let currentScale = 1;
let currentX = 0;
let currentY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;
let graphEl;
let svg;
let graphWidth;
let graphHeight;

function updateTransform() {
    svg.style.transform = `translate(${currentX}px, ${currentY}px) scale(${currentScale})`;
    svg.style.transformOrigin = '0 0';
}

function fitToScreen() {
    const container = graphEl.parentElement;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    const scaleX = containerWidth / graphWidth;
    const scaleY = containerHeight / graphHeight;
    currentScale = Math.min(scaleX, scaleY, 1) * 0.95;
    
    const scaledWidth = graphWidth * currentScale;
    const scaledHeight = graphHeight * currentScale;
    currentX = (containerWidth - scaledWidth) / 2;
    currentY = (containerHeight - scaledHeight) / 2;
    
    updateTransform();
}

function resetZoom() {
    currentScale = 1;
    const container = graphEl.parentElement;
    currentX = (container.clientWidth - graphWidth) / 2;
    currentY = (container.clientHeight - graphHeight) / 2;
    updateTransform();
}

function zoomIn() {
    const container = graphEl.parentElement;
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    
    const oldScale = currentScale;
    currentScale = Math.min(currentScale * 1.2, 3);
    const scaleRatio = currentScale / oldScale;
    
    currentX = centerX - (centerX - currentX) * scaleRatio;
    currentY = centerY - (centerY - currentY) * scaleRatio;
    
    updateTransform();
}

function zoomOut() {
    const container = graphEl.parentElement;
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    
    const oldScale = currentScale;
    currentScale = Math.max(currentScale / 1.2, 0.3);
    const scaleRatio = currentScale / oldScale;
    
    currentX = centerX - (centerX - currentX) * scaleRatio;
    currentY = centerY - (centerY - currentY) * scaleRatio;
    
    updateTransform();
}

// ==================== Mouse Wheel Zoom ====================
function setupWheelZoom() {
    graphEl.addEventListener('wheel', (e) => {
        e.preventDefault();
        
        const rect = graphEl.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const oldScale = currentScale;
        const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
        currentScale = Math.max(0.3, Math.min(3, currentScale * zoomDelta));
        
        const scaleRatio = currentScale / oldScale;
        currentX = mouseX - (mouseX - currentX) * scaleRatio;
        currentY = mouseY - (mouseY - currentY) * scaleRatio;
        
        updateTransform();
    }, { passive: false });
}

// ==================== Pan with Drag ====================
function setupPanning() {
    graphEl.addEventListener('mousedown', (e) => {
        if (e.target.closest('.node')) {
            return;
        }
        isDragging = true;
        startX = e.clientX - currentX;
        startY = e.clientY - currentY;
        graphEl.style.cursor = 'grabbing';
        e.preventDefault();
    });
    
    graphEl.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        currentX = e.clientX - startX;
        currentY = e.clientY - startY;
        updateTransform();
    });
    
    graphEl.addEventListener('mouseup', () => {
        isDragging = false;
        graphEl.style.cursor = 'grab';
    });
    
    graphEl.addEventListener('mouseleave', () => {
        isDragging = false;
        graphEl.style.cursor = 'grab';
    });
}

// ==================== Initialization ====================
function initializeGraph() {
    graphEl = document.getElementById('graph');
    
    const layoutResult = layoutGraph();
    const g = layoutResult.g;
    graphWidth = layoutResult.graphWidth;
    graphHeight = layoutResult.graphHeight;
    
    svg = createSVG(graphWidth, graphHeight);
    const defs = createMarkers(svg);
    
    renderEdges(svg, g, defs);
    renderNodes(svg, g, defs);
    
    graphEl.appendChild(svg);
    
    setupTooltips();
    setupWheelZoom();
    setupPanning();
    
    setTimeout(fitToScreen, 150);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeGraph);
} else {
    initializeGraph();
}