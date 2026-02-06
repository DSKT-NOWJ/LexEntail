#!/usr/bin/env python3
"""
Utility to find the best checkpoint from training output.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional


def find_best_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the best checkpoint from the training output directory.
    
    Args:
        output_dir: Training output directory containing checkpoints
        
    Returns:
        Path to the best checkpoint, or None if not found
    """
    output_path = Path(output_dir)
    
    # Check if output directory exists
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}", file=sys.stderr)
        return None
    
    # Method 1: Check trainer_state.json for best model checkpoint
    trainer_state_file = output_path / "trainer_state.json"
    if trainer_state_file.exists():
        try:
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            
            best_checkpoint = trainer_state.get("best_model_checkpoint")
            if best_checkpoint and Path(best_checkpoint).exists():
                return best_checkpoint
                
        except Exception as e:
            print(f"Warning: Could not read trainer_state.json: {e}", file=sys.stderr)
    
    # Method 2: Find the latest checkpoint as fallback
    checkpoint_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step_num = int(item.name.split("-")[1])
                checkpoint_dirs.append((step_num, item))
            except (ValueError, IndexError):
                continue
    
    if checkpoint_dirs:
        # Sort by step number and take the highest (latest)
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = str(checkpoint_dirs[0][1])
        print(f"Warning: Using latest checkpoint as fallback: {latest_checkpoint}", file=sys.stderr)
        return latest_checkpoint
    
    print(f"Error: No checkpoints found in {output_dir}", file=sys.stderr)
    return None


def main():
    """Command line interface."""
    if len(sys.argv) != 2:
        print("Usage: python find_best_checkpoint.py <output_dir>", file=sys.stderr)
        print("Output: prints the path to the best checkpoint to stdout", file=sys.stderr)
        sys.exit(1)
    
    output_dir = sys.argv[1]
    best_checkpoint = find_best_checkpoint(output_dir)
    
    if best_checkpoint:
        print(best_checkpoint)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()