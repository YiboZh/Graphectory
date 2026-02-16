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
    
    // Add nodes with fixed sizing (no thought-based scaling)
    nodesData.forEach(node => {
        const lines = node.label.split('\\n');
        const maxLineLength = Math.max(...lines.map(l => l.length));
        const width = Math.max(140, maxLineLength * 8);
        const height = Math.max(60, 35 + lines.length * 18);
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
    
    // CRITICAL FIX: Normalize coordinates to start at (0, 0)
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
function calculateEdgeStyle(edge) {
    /**
     * Calculate edge styling based on thought length and multi-node status
     * 
     * Rules:
     * - is_multi_node_step = true: Blue dotted line, minimum width (nodes in same step)
     * - thought_length = 0: Gray dotted line, minimum width (no thinking)
     * - thought_length > 0: Solid line, width scales 2-8px based on length (0-1000 chars)
     */
    let strokeWidth = 2;  // default
    let strokeDasharray = '';  // solid by default
    let stroke = '#95a5a6';  // default gray
    let markerEnd = 'url(#arrowhead-exec)';
    
    if (edge.type === 'exec') {
        if (edge.is_multi_node_step) {
            // Multiple nodes in same trajectory step: blue dotted, minimum size
            strokeWidth = 1;
            strokeDasharray = '5, 5';
            stroke = '#3498db';
            markerEnd = 'url(#arrowhead-exec-multi)';
        } else if (edge.thought_length === 0) {
            // No thought: gray dotted line, minimum size
            strokeWidth = 1;
            strokeDasharray = '5, 5';
            stroke = '#95a5a6';
            markerEnd = 'url(#arrowhead-exec)';
        } else {
            // Normal exec edge with thought: solid line, width based on thought length
            // Map thought_length (0-1000 chars) to stroke width (2-8px)
            const maxThought = 1000;
            const minWidth = 2;
            const maxWidth = 8;
            const normalizedThought = Math.min(edge.thought_length, maxThought);
            strokeWidth = minWidth + (normalizedThought / maxThought) * (maxWidth - minWidth);
            stroke = '#95a5a6';
            markerEnd = 'url(#arrowhead-exec)';
        }
    } else if (edge.type === 'hier') {
        // Hierarchical edge styling
        strokeWidth = 2;
        strokeDasharray = '5, 5';
        stroke = '#27ae60';
        markerEnd = 'url(#arrowhead-hier)';
    }
    
    return { strokeWidth, strokeDasharray, stroke, markerEnd };
}

function renderEdges(svg, g, defs) {
    const edgesByPair = {};
    edgesData.forEach((edge, idx) => {
        const key = `${edge.from}-${edge.to}`;
        if (!edgesByPair[key]) {
            edgesByPair[key] = [];
        }
        edgesByPair[key].push({ ...edge, idx });
    });
    
    g.edges().forEach(e => {
        const edge = g.edge(e);
        const edgeKey = `${e.v}-${e.w}`;
        const edgesInPair = edgesByPair[edgeKey] || [];
        const edgeIndex = edgesInPair.findIndex(ed => ed.type === edge.type && ed.label === edge.label);
        const totalEdges = edgesInPair.length;
        
        const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        edgeGroup.setAttribute('class', `edge ${edge.type}`);
        
        const style = calculateEdgeStyle(edge);
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const points = edge.points;
        
        let offsetY = 0;
        if (totalEdges > 1) {
            offsetY = (edgeIndex - (totalEdges - 1) / 2) * 15;
        }
        
        let d = `M ${points[0].x} ${points[0].y + offsetY}`;
        for (let i = 1; i < points.length; i++) {
            d += ` L ${points[i].x} ${points[i].y + offsetY}`;
        }
        
        path.setAttribute('d', d);
        path.setAttribute('stroke', style.stroke);
        path.setAttribute('stroke-width', style.strokeWidth);
        if (style.strokeDasharray) {
            path.setAttribute('stroke-dasharray', style.strokeDasharray);
        }
        path.setAttribute('marker-end', style.markerEnd);
        path.setAttribute('fill', 'none');
        
        edgeGroup.appendChild(path);
        
        if (edge.label) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            const midPoint = points[Math.floor(points.length / 2)];
            text.setAttribute('x', midPoint.x);
            text.setAttribute('y', midPoint.y + offsetY - 5);
            text.setAttribute('text-anchor', 'middle');
            text.textContent = edge.label;
            edgeGroup.appendChild(text);
        }
        
        svg.appendChild(edgeGroup);
    });
}

// ==================== Node Rendering ====================
function renderNodes(svg, g, defs) {
    g.nodes().forEach(nodeId => {
        const node = g.node(nodeId);
        const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodeGroup.setAttribute('class', 'node');
        nodeGroup.setAttribute('data-id', nodeId);
        nodeGroup.setAttribute('data-tooltip', node.tooltip);
        
        if (node.colors.length > 1) {
            const gradId = `grad-${nodeId.replace(/[^a-zA-Z0-9]/g, '')}`;
            const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            grad.setAttribute('id', gradId);
            grad.setAttribute('x1', '0%');
            grad.setAttribute('y1', '0%');
            grad.setAttribute('x2', '100%');
            grad.setAttribute('y2', '0%');
            
            node.colors.forEach((color, i) => {
                const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                stop1.setAttribute('offset', `${i / node.colors.length * 100}%`);
                stop1.setAttribute('stop-color', color);
                grad.appendChild(stop1);
                
                const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                stop2.setAttribute('offset', `${(i + 1) / node.colors.length * 100}%`);
                stop2.setAttribute('stop-color', color);
                grad.appendChild(stop2);
            });
            
            defs.appendChild(grad);
            
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', node.x - node.width / 2);
            rect.setAttribute('y', node.y - node.height / 2);
            rect.setAttribute('width', node.width);
            rect.setAttribute('height', node.height);
            rect.setAttribute('fill', `url(#${gradId})`);
            
            // Add thick red border for failed actions
            if (node.has_failure) {
                rect.setAttribute('stroke', '#e74c3c');
                rect.setAttribute('stroke-width', '4');
            }
            
            nodeGroup.appendChild(rect);
        } else {
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', node.x - node.width / 2);
            rect.setAttribute('y', node.y - node.height / 2);
            rect.setAttribute('width', node.width);
            rect.setAttribute('height', node.height);
            rect.setAttribute('fill', node.color);
            
            // Add thick red border for failed actions
            if (node.has_failure) {
                rect.setAttribute('stroke', '#e74c3c');
                rect.setAttribute('stroke-width', '4');
            }
            
            nodeGroup.appendChild(rect);
        }
        
        const lines = node.label.split('\\n');
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
                text.setAttribute('font-weight', 'bold');
                text.setAttribute('font-size', '14');
            } else {
                text.setAttribute('font-size', '11');
                text.setAttribute('fill', '#555');
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