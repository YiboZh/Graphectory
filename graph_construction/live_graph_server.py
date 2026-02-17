#!/usr/bin/env python3
"""
Live Trajectory Graph Server

A web server that generates and renders trajectory graphs on-demand.
No pre-generation of HTML files required - all rendering happens live.

Usage:
    python live_graph_server.py --graphs_dir path/to/graphs --eval_report report.json --port 8000
"""

import argparse
import os
import json
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from buildGraph import GraphBuilder, build_hierarchical_edges, determine_resolution_status
from visualizer import GraphVisualizer


class LiveGraphHandler(BaseHTTPRequestHandler):
    """HTTP handler for live graph generation and rendering."""
    
    # Class variables set by main()
    graphs_dir = None
    eval_report_path = None
    parser = None
    template_dir = None
    
    def log_message(self, format, *args):
        """Override to provide cleaner logging."""
        pass  # Suppress default logging
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        try:
            # Serve index page
            if path == '/' or path == '/index.html':
                self.serve_index()
            # Serve graph list API
            elif path == '/api/graphs':
                self.serve_graph_list()
            # Serve live graph rendering
            elif path == '/api/graph':
                instance_id = query_params.get('id', [''])[0]
                filter_cd = query_params.get('filter_cd', ['true'])[0].lower() == 'true'
                if instance_id:
                    self.serve_live_graph(instance_id, filter_cd)
                else:
                    self.send_error(400, "Missing instance_id parameter")
            # Serve static assets
            elif path == '/graph_renderer.js':
                self.serve_file('graph_renderer.js', 'text/javascript')
            elif path == '/styles.css':
                self.serve_file('styles.css', 'text/css')
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            print(f"[ERROR] {path}: {str(e)}")
            traceback.print_exc()
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def serve_file(self, filename, content_type):
        """Serve a static file from the template directory."""
        file_path = self.template_dir / filename
        if file_path.exists():
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File not found: {filename}")
    
    def serve_index(self):
        """Serve the main index page with graph browser."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Trajectory Graph Browser</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            bottom: 0;
            width: 350px;
            background: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .header h1 {
            font-size: 20px;
            margin-bottom: 8px;
        }
        
        .header .subtitle {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .controls {
            padding: 16px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .search-input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            margin-bottom: 12px;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .toggle-container {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
        }
        
        .toggle {
            position: relative;
            width: 50px;
            height: 26px;
        }
        
        .toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            border-radius: 26px;
            transition: .4s;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            border-radius: 50%;
            transition: .4s;
        }
        
        .toggle input:checked + .toggle-slider {
            background-color: #667eea;
        }
        
        .toggle input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        
        .toggle-label {
            font-size: 13px;
            color: #555;
        }
        
        .stats {
            padding: 12px 16px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            font-size: 12px;
            color: #666;
        }
        
        .graph-list {
            flex: 1;
            overflow-y: auto;
        }
        
        .graph-item {
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .graph-item:hover {
            background: #f8f9fa;
        }
        
        .graph-item.active {
            background: #e3e7ff;
            border-left: 4px solid #667eea;
        }
        
        .graph-item-title {
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
        }
        
        .graph-item-meta {
            font-size: 12px;
            color: #7f8c8d;
            display: flex;
            gap: 12px;
        }
        
        .badge {
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .badge-resolved {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-unresolved {
            background: #f8d7da;
            color: #721c24;
        }
        
        .main-content {
            margin-left: 350px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .graph-container {
            flex: 1;
            background: white;
            position: relative;
        }
        
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #7f8c8d;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .no-selection {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #7f8c8d;
        }
        
        .no-selection-icon {
            font-size: 64px;
            margin-bottom: 16px;
            opacity: 0.3;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="header">
            <h1>📊 Live Graph Browser</h1>
            <div class="subtitle">On-demand rendering</div>
        </div>
        
        <div class="controls">
            <input 
                type="text" 
                class="search-input" 
                id="searchInput" 
                placeholder="Search instances..."
                autofocus
            >
            
            <div class="toggle-container">
                <label class="toggle">
                    <input type="checkbox" id="filterCdToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
                <span class="toggle-label">Filter cd commands (show ▲ hat)</span>
            </div>
        </div>
        
        <div class="stats" id="stats">Loading...</div>
        
        <div class="graph-list" id="graphList">
            <div class="loading">
                <div class="loading-spinner"></div>
                Loading graphs...
            </div>
        </div>
    </div>
    
    <div class="main-content">
        <div class="graph-container" id="graphContainer">
            <div class="no-selection">
                <div class="no-selection-icon">📊</div>
                <div>Select a graph from the list to view</div>
            </div>
        </div>
    </div>
    
    <script>
        let allGraphs = [];
        let currentInstanceId = null;
        
        // Load graph list
        async function loadGraphList() {
            try {
                const response = await fetch('/api/graphs');
                allGraphs = await response.json();
                updateStats();
                displayGraphList(allGraphs);
            } catch (error) {
                document.getElementById('graphList').innerHTML = 
                    '<div style="padding: 20px; color: #e74c3c;">Error loading graphs</div>';
            }
        }
        
        // Update statistics
        function updateStats() {
            const total = allGraphs.length;
            const resolved = allGraphs.filter(g => g.status === 'resolved').length;
            const unresolved = allGraphs.filter(g => g.status === 'unresolved').length;
            
            document.getElementById('stats').innerHTML = 
                `Total: ${total} | Resolved: ${resolved} | Unresolved: ${unresolved}`;
        }
        
        // Display graph list
        function displayGraphList(graphs) {
            const listEl = document.getElementById('graphList');
            
            if (graphs.length === 0) {
                listEl.innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No graphs found</div>';
                return;
            }
            
            listEl.innerHTML = graphs.map(graph => `
                <div class="graph-item" data-id="${graph.instance_id}" onclick="loadGraph('${graph.instance_id}')">
                    <div class="graph-item-title">${graph.instance_id}</div>
                    <div class="graph-item-meta">
                        <span class="badge badge-${graph.status}">${graph.status}</span>
                        <span>Difficulty: ${graph.difficulty}</span>
                    </div>
                </div>
            `).join('');
        }
        
        // Search graphs
        document.getElementById('searchInput').addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = allGraphs.filter(g => 
                g.instance_id.toLowerCase().includes(query)
            );
            displayGraphList(filtered);
        });
        
        // Load graph on-demand
        async function loadGraph(instanceId) {
            currentInstanceId = instanceId;
            
            // Update active state
            document.querySelectorAll('.graph-item').forEach(item => {
                item.classList.toggle('active', item.dataset.id === instanceId);
            });
            
            const container = document.getElementById('graphContainer');
            container.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    Generating graph...
                </div>
            `;
            
            try {
                const filterCd = document.getElementById('filterCdToggle').checked;
                const response = await fetch(`/api/graph?id=${encodeURIComponent(instanceId)}&filter_cd=${filterCd}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const html = await response.text();
                container.innerHTML = html;
            } catch (error) {
                container.innerHTML = `
                    <div class="loading">
                        <div style="color: #e74c3c;">❌ Error loading graph</div>
                        <div style="font-size: 14px; margin-top: 8px;">${error.message}</div>
                    </div>
                `;
            }
        }
        
        // Reload graph when cd filter toggle changes
        document.getElementById('filterCdToggle').addEventListener('change', () => {
            if (currentInstanceId) {
                loadGraph(currentInstanceId);
            }
        });
        
        // Load initial list
        loadGraphList();
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_graph_list(self):
        """Serve list of all available graphs as JSON."""
        graphs = []
        
        # Find all JSON files
        for root, dirs, files in os.walk(self.graphs_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = Path(root) / file
                    
                    try:
                        with open(json_path, 'r') as f:
                            graph_data = json.load(f)
                        
                        instance_id = graph_data.get('graph', {}).get('instance_name', file.replace('.json', ''))
                        status = graph_data.get('graph', {}).get('resolution_status', 'unknown')
                        difficulty = graph_data.get('graph', {}).get('debug_difficulty', 'unknown')
                        
                        # Check if trajectory file exists
                        traj_path = json_path.parent / f"{instance_id}.traj"
                        if traj_path.exists():
                            graphs.append({
                                'instance_id': instance_id,
                                'status': status,
                                'difficulty': difficulty,
                                'json_path': str(json_path),
                                'traj_path': str(traj_path)
                            })
                    except Exception as e:
                        print(f"[WARN] Error reading {json_path}: {e}")
                        continue
        
        # Sort by instance_id
        graphs.sort(key=lambda x: x['instance_id'])
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(graphs).encode())
    
    def serve_live_graph(self, instance_id, filter_cd):
        """Generate and serve a graph on-demand."""
        try:
            # Find trajectory file
            traj_path = None
            for root, dirs, files in os.walk(self.graphs_dir):
                target_file = f"{instance_id}.traj"
                if target_file in files:
                    traj_path = Path(root) / target_file
                    break
            
            if not traj_path or not traj_path.exists():
                self.send_error(404, f"Trajectory not found: {instance_id}")
                return
            
            # Load trajectory
            with open(traj_path, 'r') as f:
                traj_data = json.load(f)
            
            # Build graph with cd filtering option
            graph = self.build_graph_from_trajectory(
                traj_data, 
                instance_id, 
                filter_cd=filter_cd
            )
            
            # Generate HTML
            html = self.generate_graph_html(graph, instance_id, filter_cd)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(html.encode())
            
        except Exception as e:
            print(f"[ERROR] Failed to generate graph for {instance_id}: {e}")
            traceback.print_exc()
            self.send_error(500, f"Graph generation failed: {str(e)}")
    
    def build_graph_from_trajectory(self, traj_data, instance_id, filter_cd=True):
        """Build graph from trajectory data with optional cd filtering."""
        from mapPhase import get_phase
        
        builder = GraphBuilder()
        trajectory = traj_data.get("trajectory", [])
        
        for step_idx, step in enumerate(trajectory):
            action_str = step.get("action", "")
            thought = step.get("thought", "") or ""
            thought_length = len(thought)
            
            # Handle explicit "think" steps
            if action_str.strip() == "":
                node_key = builder.add_or_update_node(
                    node_label="think",
                    args={"thought_len": thought_length},
                    flags={},
                    phase="general",
                    step_idx=step_idx,
                    tool=None,
                    command=None,
                    subcommand=None,
                    thought_length=thought_length,
                    has_cd=False
                )
                builder.add_execution_edge(node_key, step_idx, is_first_in_step=True)
                builder.update_previous_node(node_key)
                builder.add_phase("general")
                continue
            
            # Parse commands
            parsed_commands = self.parser.parse(action_str)
            if not parsed_commands:
                continue
            
            # Apply cd filtering based on toggle
            has_cd = False
            filtered_commands = []
            
            if filter_cd and len(parsed_commands) > 1:
                first_cmd = parsed_commands[0]
                if first_cmd.get("command", "").strip().lower() == "cd":
                    has_cd = True
                    filtered_commands = parsed_commands[1:]
                else:
                    filtered_commands = parsed_commands
            else:
                filtered_commands = parsed_commands
            
            # Track first edge
            is_first_in_step = True
            
            for parsed in filtered_commands:
                tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
                subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
                command = parsed.get("command", "").strip() if parsed.get("command") else ""
                args = parsed.get("args", {})
                flags = parsed.get("flags", {})
                
                if tool:
                    node_label = f"{tool}: {subcommand}" if subcommand else tool
                else:
                    node_label = command.strip() or action_str.strip()
                
                phase = get_phase(tool, subcommand, command, args, builder.prev_phases)
                
                # Check edit status
                from buildGraph import check_edit_status
                edit_status = check_edit_status(tool, subcommand, args, step.get("observation", ""))
                if edit_status and isinstance(args, dict):
                    args["edit_status"] = edit_status
                
                node_key = builder.add_or_update_node(
                    node_label=node_label,
                    args=args,
                    flags=flags,
                    phase=phase,
                    step_idx=step_idx,
                    tool=tool,
                    command=command,
                    subcommand=subcommand,
                    thought_length=thought_length,
                    has_cd=has_cd
                )
                
                builder.add_execution_edge(node_key, step_idx, is_first_in_step=is_first_in_step)
                builder.update_previous_node(node_key)
                builder.add_phase(phase)
                
                is_first_in_step = False
        
        # Build hierarchical edges
        build_hierarchical_edges(builder.G, builder.localization_nodes)
        
        # Add metadata
        resolution_status = determine_resolution_status(instance_id, self.eval_report_path)
        builder.G.graph["resolution_status"] = resolution_status
        builder.G.graph["instance_name"] = instance_id
        
        # Try to get difficulty from various sources
        try:
            from buildGraph import difficulty_lookup
            builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
        except:
            builder.G.graph["debug_difficulty"] = "unknown"
        
        return builder.G
    
    def generate_graph_html(self, G, instance_id, filter_cd):
        """Generate HTML for the graph."""
        visualizer = GraphVisualizer(template_dir=self.template_dir)
        
        # Prepare data
        nodes_data = visualizer._prepare_nodes_data(G)
        edges_data = visualizer._prepare_edges_data(G)
        
        # Get metadata
        instance_name = G.graph.get("instance_name", instance_id)
        resolution_status = G.graph.get("resolution_status", "unknown")
        difficulty = G.graph.get("debug_difficulty", "unknown")
        
        # Build metadata comment
        mode = "CD filtered (hat mode)" if filter_cd else "CD as separate node"
        metadata_comment = f"Rendering mode: {mode}"
        
        # Load template
        template_path = self.template_dir / "graph_template.html"
        with open(template_path, 'r') as f:
            html_template = f.read()
        
        # Load CSS
        css_path = self.template_dir / "styles.css"
        with open(css_path, 'r') as f:
            css_content = f.read()
        
        # Load JS
        js_path = self.template_dir / "graph_renderer.js"
        with open(js_path, 'r') as f:
            js_content = f.read()
        
        # Replace placeholders
        html = html_template.replace("{{INSTANCE_NAME}}", visualizer._escape_html(instance_name))
        html = html.replace("{{RESOLUTION_STATUS}}", resolution_status)
        html = html.replace("{{DIFFICULTY}}", visualizer._escape_html(str(difficulty)))
        html = html.replace("{{NODE_COUNT}}", str(len(nodes_data)))
        html = html.replace("{{EDGE_COUNT}}", str(len(edges_data)))
        html = html.replace("{{METADATA_COMMENT}}", visualizer._escape_html(metadata_comment))
        html = html.replace("{{NODES_DATA}}", json.dumps(nodes_data))
        html = html.replace("{{EDGES_DATA}}", json.dumps(edges_data))
        html = html.replace("{{PHASE_COLORS}}", json.dumps(visualizer.phase_colors))
        
        # Inline CSS and JS
        html = html.replace('<link rel="stylesheet" href="styles.css">', f'<style>{css_content}</style>')
        html = html.replace('<script src="graph_renderer.js"></script>', f'<script>{js_content}</script>')
        
        return html


def setup_parser():
    """Setup CommandParser with tool configurations."""
    try:
        from commandParser import CommandParser
        parser = CommandParser()
        
        # Try to load tool configs if they exist
        tool_configs = [
            "data/SWE-agent/tools/edit_anthropic/config.yaml",
            "data/SWE-agent/tools/review_on_submit_m/config.yaml",
            "data/SWE-agent/tools/registry/config.yaml",
        ]
        
        for config in tool_configs:
            if Path(config).exists():
                parser.load_tool_yaml_files([config])
                break
        
        return parser
    except ImportError:
        print("[WARN] CommandParser not available, using basic parsing")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live trajectory graph server with on-demand rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with trajectory directory
  python live_graph_server.py --graphs_dir output/trajectories --eval_report report.json --port 8000
  
  # With custom template directory
  python live_graph_server.py --graphs_dir trajectories --eval_report report.json \\
      --template_dir custom_templates --port 8080

Then open http://localhost:8000 in your browser.

Features:
  - On-demand graph generation (no pre-generation needed)
  - Toggle cd filtering on/off in real-time
  - Search and filter instances
  - Live statistics
        """
    )
    
    parser.add_argument(
        '--graphs_dir',
        type=str,
        required=True,
        help='Directory containing trajectory .traj and .json files'
    )
    
    parser.add_argument(
        '--eval_report',
        type=str,
        required=True,
        help='Path to evaluation report JSON file'
    )
    
    parser.add_argument(
        '--template_dir',
        type=str,
        default=None,
        help='Directory containing templates (default: same as script)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run server on (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"[ERROR] Graphs directory does not exist: {graphs_dir}")
        return 1
    
    eval_report_path = Path(args.eval_report)
    if not eval_report_path.exists():
        print(f"[ERROR] Evaluation report does not exist: {eval_report_path}")
        return 1
    
    # Setup template directory
    if args.template_dir:
        template_dir = Path(args.template_dir)
    else:
        template_dir = Path(__file__).parent
    
    if not template_dir.exists():
        print(f"[ERROR] Template directory does not exist: {template_dir}")
        return 1
    
    # Setup command parser
    cmd_parser = setup_parser()
    if cmd_parser is None:
        print("[WARN] Running without command parser - limited functionality")
    
    # Set class variables
    LiveGraphHandler.graphs_dir = graphs_dir
    LiveGraphHandler.eval_report_path = str(eval_report_path)
    LiveGraphHandler.parser = cmd_parser
    LiveGraphHandler.template_dir = template_dir
    
    # Create and start server
    server_address = ('', args.port)
    httpd = HTTPServer(server_address, LiveGraphHandler)
    
    print(f"\n{'='*70}")
    print(f"Live Trajectory Graph Server")
    print(f"{'='*70}")
    print(f"Graphs directory: {graphs_dir.absolute()}")
    print(f"Eval report:      {eval_report_path.absolute()}")
    print(f"Template dir:     {template_dir.absolute()}")
    print(f"Server URL:       http://localhost:{args.port}")
    print(f"{'='*70}\n")
    print("Features:")
    print("  ✓ On-demand graph generation")
    print("  ✓ Toggle cd filtering in real-time")
    print("  ✓ No pre-generation required")
    print(f"\n{'='*70}\n")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
        return 0


if __name__ == '__main__':
    sys.exit(main())
