#!/usr/bin/env python3
"""
Trajectory Graph Web Server

A local web server for browsing and searching trajectory graphs.
Provides a clean interface with search functionality to navigate between HTMLs.

Usage:
    python graph_server.py --graphs_dir path/to/graphs --port 8000
"""

import argparse
import os
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import mimetypes


class GraphHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving trajectory graphs with search."""
    
    graphs_dir = None  # Set by main()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Serve index page
        if path == '/' or path == '/index.html':
            self.serve_index()
        # Serve search API
        elif path == '/api/search':
            query_components = parse_qs(parsed_path.query)
            search_query = query_components.get('q', [''])[0]
            self.serve_search_results(search_query)
        # Serve graph listing API
        elif path == '/api/graphs':
            self.serve_graph_list()
        # Serve specific graph HTML
        elif path.startswith('/graph/'):
            instance_id = path[7:]  # Remove '/graph/' prefix
            self.serve_graph(instance_id)
        # Serve static files
        else:
            super().do_GET()
    
    def serve_index(self):
        """Serve the main index page with search interface."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trajectory Graph Browser</title>
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
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 32px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 16px;
        }
        
        .search-box {
            background: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .search-input {
            width: 100%;
            padding: 16px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            transition: all 0.3s;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .results {
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .result-item {
            padding: 20px 24px;
            border-bottom: 1px solid #f0f0f0;
            transition: background 0.2s;
            cursor: pointer;
        }
        
        .result-item:hover {
            background: #f8f9fa;
        }
        
        .result-item:last-child {
            border-bottom: none;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        
        .result-meta {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            font-size: 14px;
            color: #7f8c8d;
        }
        
        .result-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-resolved {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-unresolved {
            background: #f8d7da;
            color: #721c24;
        }
        
        .badge-unsubmitted {
            background: #fff3cd;
            color: #856404;
        }
        
        .no-results {
            padding: 60px 24px;
            text-align: center;
            color: #7f8c8d;
            font-size: 16px;
        }
        
        .loading {
            padding: 40px 24px;
            text-align: center;
            color: #7f8c8d;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 16px;
            flex-wrap: wrap;
        }
        
        .stat-item {
            background: #f8f9fa;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .stat-label {
            color: #7f8c8d;
            margin-right: 8px;
        }
        
        .stat-value {
            color: #2c3e50;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <span>📊</span>
                Trajectory Graph Browser
            </h1>
            <p class="subtitle">Search and navigate trajectory graphs</p>
            <div class="stats" id="stats"></div>
        </div>
        
        <div class="search-box">
            <input 
                type="text" 
                class="search-input" 
                id="searchInput" 
                placeholder="Search by instance ID (e.g., django__django-12345)..."
                autofocus
            >
        </div>
        
        <div class="results" id="results">
            <div class="loading">Loading graphs...</div>
        </div>
    </div>
    
    <script>
        let allGraphs = [];
        
        // Load all graphs on page load
        async function loadGraphs() {
            try {
                const response = await fetch('/api/graphs');
                allGraphs = await response.json();
                updateStats();
                displayResults(allGraphs);
            } catch (error) {
                document.getElementById('results').innerHTML = 
                    '<div class="no-results">Error loading graphs: ' + error.message + '</div>';
            }
        }
        
        // Update statistics
        function updateStats() {
            const total = allGraphs.length;
            const resolved = allGraphs.filter(g => g.status === 'resolved').length;
            const unresolved = allGraphs.filter(g => g.status === 'unresolved').length;
            
            document.getElementById('stats').innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Total:</span>
                    <span class="stat-value">${total}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Resolved:</span>
                    <span class="stat-value">${resolved}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Unresolved:</span>
                    <span class="stat-value">${unresolved}</span>
                </div>
            `;
        }
        
        // Search graphs
        function searchGraphs(query) {
            if (!query.trim()) {
                return allGraphs;
            }
            
            const lowerQuery = query.toLowerCase();
            return allGraphs.filter(graph => 
                graph.instance_id.toLowerCase().includes(lowerQuery)
            );
        }
        
        // Display results
        function displayResults(graphs) {
            const resultsDiv = document.getElementById('results');
            
            if (graphs.length === 0) {
                resultsDiv.innerHTML = '<div class="no-results">No graphs found matching your search.</div>';
                return;
            }
            
            const html = graphs.map(graph => `
                <div class="result-item" onclick="openGraph('${graph.instance_id}')">
                    <div class="result-title">${graph.instance_id}</div>
                    <div class="result-meta">
                        <span class="result-badge badge-${graph.status}">${graph.status}</span>
                        <span>Difficulty: ${graph.difficulty}</span>
                        <span>Nodes: ${graph.node_count}</span>
                        <span>Edges: ${graph.edge_count}</span>
                    </div>
                </div>
            `).join('');
            
            resultsDiv.innerHTML = html;
        }
        
        // Open graph in new tab
        function openGraph(instanceId) {
            window.open(`/graph/${instanceId}`, '_blank');
        }
        
        // Handle search input
        document.getElementById('searchInput').addEventListener('input', (e) => {
            const query = e.target.value;
            const results = searchGraphs(query);
            displayResults(results);
        });
        
        // Load graphs on page load
        loadGraphs();
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_graph_list(self):
        """Serve list of all available graphs as JSON."""
        graphs = []
        
        # Recursively find all HTML files
        for root, dirs, files in os.walk(self.graphs_dir):
            for file in files:
                if file.endswith('.html'):
                    html_path = Path(root) / file
                    json_path = html_path.with_suffix('.json')
                    
                    # Extract metadata from JSON if available
                    metadata = {
                        'instance_id': file.replace('.html', ''),
                        'status': 'unknown',
                        'difficulty': 'unknown',
                        'node_count': 0,
                        'edge_count': 0
                    }
                    
                    if json_path.exists():
                        try:
                            with open(json_path, 'r') as f:
                                graph_data = json.load(f)
                                metadata['status'] = graph_data.get('graph', {}).get('resolution_status', 'unknown')
                                metadata['difficulty'] = graph_data.get('graph', {}).get('debug_difficulty', 'unknown')
                                metadata['node_count'] = len(graph_data.get('nodes', []))
                                metadata['edge_count'] = len(graph_data.get('edges', []))
                        except Exception:
                            pass
                    
                    graphs.append(metadata)
        
        # Sort by instance_id
        graphs.sort(key=lambda x: x['instance_id'])
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(graphs).encode())
    
    def serve_search_results(self, query):
        """Serve search results as JSON."""
        # Load all graphs and filter by query
        graphs = []
        
        for root, dirs, files in os.walk(self.graphs_dir):
            for file in files:
                if file.endswith('.html'):
                    instance_id = file.replace('.html', '')
                    
                    if query.lower() in instance_id.lower():
                        graphs.append({
                            'instance_id': instance_id,
                            'path': f"/graph/{instance_id}"
                        })
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(graphs).encode())
    
    def serve_graph(self, instance_id):
        """Serve a specific graph HTML file."""
        # Find the HTML file
        html_path = None
        for root, dirs, files in os.walk(self.graphs_dir):
            target_file = f"{instance_id}.html"
            if target_file in files:
                html_path = Path(root) / target_file
                break
        
        if html_path and html_path.exists():
            # Serve the HTML file
            with open(html_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"Graph not found: {instance_id}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Local web server for browsing trajectory graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port 8000
  python graph_server.py --graphs_dir output/SWE-agent/graphs/deepseek-v3
  
  # Start server on custom port
  python graph_server.py --graphs_dir output/graphs --port 8080

Then open http://localhost:8000 in your browser.
        """
    )
    
    parser.add_argument(
        '--graphs_dir',
        type=str,
        required=True,
        help='Directory containing graph HTML files'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run server on (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Validate graphs directory
    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"Error: Graphs directory does not exist: {graphs_dir}")
        return 1
    
    if not graphs_dir.is_dir():
        print(f"Error: Path is not a directory: {graphs_dir}")
        return 1
    
    # Set graphs directory for handler
    GraphHTTPHandler.graphs_dir = graphs_dir
    
    # Change to graphs directory so relative paths work
    os.chdir(graphs_dir)
    
    # Create and start server
    server_address = ('', args.port)
    httpd = HTTPServer(server_address, GraphHTTPHandler)
    
    print(f"\n{'='*60}")
    print(f"Trajectory Graph Server")
    print(f"{'='*60}")
    print(f"Serving graphs from: {graphs_dir.absolute()}")
    print(f"Server running at:   http://localhost:{args.port}")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
        return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
