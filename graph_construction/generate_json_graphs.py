#!/usr/bin/env python3
"""
JSON-Only Graph Generation Script

This script generates trajectory graph JSON files (no HTML).
Use live_graph_server.py to view graphs with on-demand rendering.

Usage:
    python generate_json_graphs.py --agent sa --model dsk-v3 --trajs path/to/trajs --eval_report report.json --output_dir output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from buildGraph import GraphBuilder, build_hierarchical_edges, determine_resolution_status


# ==================== Configuration ====================
SUPPORTED_AGENTS = {"sa", "oh"}
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4", "gpt-5m", "dsk-v3.2"}

MODEL_NAMES = {
    "dsk-v3": "deepseek-v3",
    "dsk-r1": "deepseek-r1-0528",
    "dev": "devstral-small",
    "cld-4": "claude-sonnet-4",
    "gpt-5m": "gpt5-mini",
    "dsk-v3.2": "deepseek-v3.2"
}

AGENT_NAMES = {
    "sa": "SWE-agent",
    "oh": "OpenHands"
}


@dataclass
class ProcessingResult:
    """Result of processing a single trajectory."""
    instance_id: str
    status: str
    json_path: str = None
    error: str = None


def get_graph_output_dir(base_output_dir: str, agent: str, model: str) -> Path:
    """Construct the graph output directory path."""
    agent_name = AGENT_NAMES[agent]
    model_name = MODEL_NAMES[model]
    return Path(base_output_dir) / agent_name / "graphs" / model_name


class TrajectoryLoader:
    """Load agent trajectories."""
    
    @staticmethod
    def load_sa_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load SWE-agent trajectories from directory structure."""
        trajectories = []
        
        if not trajs_path.is_dir():
            raise ValueError(f"SA trajectories path must be a directory: {trajs_path}")
        
        for instance_dir in sorted(trajs_path.iterdir()):
            if not instance_dir.is_dir():
                continue
            
            instance_id = instance_dir.name
            traj_file = instance_dir / f"{instance_id}.traj"
            
            if not traj_file.exists():
                print(f"[WARN] Missing .traj file for {instance_id}, skipping")
                continue
            
            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse {traj_file}: {e}")
                continue
        
        return trajectories


def save_trajectory_json(traj_data: Dict, output_dir: Path, instance_id: str):
    """Save trajectory JSON file."""
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = instance_dir / f"{instance_id}.json"
    traj_path = instance_dir / f"{instance_id}.traj"
    
    # Save trajectory data
    with open(traj_path, 'w') as f:
        json.dump(traj_data, f, indent=2)
    
    # Create placeholder JSON with metadata
    metadata = {
        "graph": {
            "instance_name": instance_id,
            "resolution_status": "unknown",
            "debug_difficulty": "unknown"
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(json_path)


def process_trajectory(instance_id: str, traj_data: Dict, output_dir: Path) -> ProcessingResult:
    """Process a single trajectory and save JSON."""
    try:
        json_path = save_trajectory_json(traj_data, output_dir, instance_id)
        
        return ProcessingResult(
            instance_id=instance_id,
            status="success",
            json_path=json_path
        )
    except Exception as e:
        return ProcessingResult(
            instance_id=instance_id,
            status="error",
            error=str(e)
        )


def process_batch(trajectories: List[Dict], output_dir: Path, max_workers: int = 8) -> Dict[str, List]:
    """Process trajectories in parallel."""
    results = {"success": [], "failed": []}
    
    total = len(trajectories)
    print(f"\n{'='*70}")
    print(f"Processing {total} trajectories with {max_workers} workers...")
    print(f"{'='*70}\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(
                process_trajectory,
                traj["instance_id"],
                traj["traj_data"],
                output_dir
            ): traj["instance_id"]
            for traj in trajectories
        }
        
        completed = 0
        for future in as_completed(future_to_instance):
            result = future.result()
            completed += 1
            
            if result.status == "success":
                results["success"].append(result)
                print(f"[{completed}/{total}] ✓ {result.instance_id}")
            else:
                results["failed"].append(result)
                print(f"[{completed}/{total}] ✗ {result.instance_id}: {result.error}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate trajectory JSON files (no HTML generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate JSON files
  python generate_json_graphs.py --agent sa --model dsk-v3 \\
      --trajs trajectories/ --eval_report report.json --output_dir output

  # Then start live server
  python live_graph_server.py --graphs_dir output/SWE-agent/graphs/deepseek-v3 \\
      --eval_report report.json --port 8000
        """
    )
    
    parser.add_argument("--agent", type=str, required=True, choices=list(SUPPORTED_AGENTS))
    parser.add_argument("--model", type=str, required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--trajs", type=str, required=True)
    parser.add_argument("--eval_report", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    
    # Validate paths
    trajs_path = Path(args.trajs)
    eval_report_path = Path(args.eval_report)
    
    if not trajs_path.exists():
        print(f"[ERROR] Trajectories path does not exist: {trajs_path}")
        sys.exit(1)
    
    if not eval_report_path.exists():
        print(f"[ERROR] Evaluation report does not exist: {eval_report_path}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = get_graph_output_dir(args.output_dir, args.agent, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[args.agent]}")
    print(f"Model:        {MODEL_NAMES[args.model]}")
    print(f"Trajectories: {trajs_path}")
    print(f"Output:       {output_dir}")
    print(f"Workers:      {args.workers}")
    print(f"{'='*70}\n")
    
    # Load trajectories
    print("Loading trajectories...")
    try:
        if args.agent == "sa":
            trajectories = TrajectoryLoader.load_sa_trajectories(trajs_path)
        else:
            print(f"[ERROR] Agent '{args.agent}' not yet implemented")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load trajectories: {e}")
        sys.exit(1)
    
    if not trajectories:
        print("[ERROR] No trajectories found")
        sys.exit(1)
    
    print(f"Loaded {len(trajectories)} trajectories\n")
    
    # Process trajectories
    results = process_batch(trajectories, output_dir, max_workers=args.workers)
    
    # Print summary
    success_count = len(results["success"])
    failed_count = len(results["failed"])
    total = success_count + failed_count
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total:    {total}")
    print(f"Success:  {success_count}")
    print(f"Failed:   {failed_count}")
    print(f"{'='*70}\n")
    
    if success_count > 0:
        print("✓ JSON files generated successfully!")
        print(f"\n💡 Start the live server:")
        print(f"   python live_graph_server.py --graphs_dir {output_dir} \\")
        print(f"       --eval_report {eval_report_path} --port 8000\n")
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
