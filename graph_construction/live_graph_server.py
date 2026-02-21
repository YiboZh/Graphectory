#!/usr/bin/env python3
"""
live_graph_server.py

Single entry point.  Run:

    python live_graph_server.py --graphs_dir <dir> --eval_report <file>

Then open http://localhost:8000 in your browser.

All graph data is rendered on the fly; no HTML files are pre-generated.
Use the toggle in the sidebar to switch between cd-filtered (▲ hat) and
cd-as-separate-node mode in real time.
"""

from commandParser import CommandParser
import argparse
import sys
from http.server import HTTPServer
from pathlib import Path

# Allow sibling imports (buildGraph, mapPhase, commandParser…)
sys.path.insert(0, str(Path(__file__).parent))

from server.handler import GraphHandler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live trajectory graph browser (on-demand rendering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python live_graph_server.py \\
      --graphs_dir output/SWE-agent/graphs/deepseek-v3 \\
      --eval_report report.json

  python live_graph_server.py \\
      --graphs_dir trajectories \\
      --eval_report report.json \\
      --assets_dir custom_templates \\
      --port 8080
        """,
    )
    p.add_argument("--graphs_dir",    required=True,
                   help="Directory that contains .traj files")
    p.add_argument("--eval_report",   required=True,
                   help="Evaluation report JSON used for resolution status")
    p.add_argument("--assets_dir",    default=None,
                   help="Directory with graph_template.html / styles.css / "
                        "graph_renderer.js  (defaults to same dir as this script)")
    p.add_argument("--port",          type=int, default=8000)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"[ERROR] graphs_dir does not exist: {graphs_dir}")
        return 1

    eval_report = Path(args.eval_report)
    if not eval_report.exists():
        print(f"[ERROR] eval_report does not exist: {eval_report}")
        return 1

    assets_dir = Path(args.assets_dir) if args.assets_dir else Path(__file__).parent
    if not assets_dir.exists():
        print(f"[ERROR] assets_dir does not exist: {assets_dir}")
        return 1

    # Inject configuration into the handler class
    GraphHandler.graphs_dir       = graphs_dir
    GraphHandler.eval_report_path = str(eval_report)
    GraphHandler.cmd_parser       = CommandParser()
    GraphHandler.assets_dir       = assets_dir

    httpd = HTTPServer(("", args.port), GraphHandler)

    print(f"\n{'─'*60}")
    print(f"  Trajectory Graph Server")
    print(f"{'─'*60}")
    print(f"  Graphs dir   : {graphs_dir.absolute()}")
    print(f"  Eval report  : {eval_report.absolute()}")
    print(f"  Assets dir   : {assets_dir.absolute()}")
    print(f"  Tool configs : auto-discovered from each instance folder")
    print(f"                 (<graphs_dir>/<id>/<id>.config.yaml)")
    print(f"  URL          : http://localhost:{args.port}")
    print(f"{'─'*60}\n")
    print("  Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

    return 0


if __name__ == "__main__":
    sys.exit(main())