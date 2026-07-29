# cot_extraction/run_cot_yeto.py
#!/usr/bin/env python3
"""Integrated CoT extraction + Yeto fine-tuning using local credentials."""
import subprocess
import sys
import argparse
from pathlib import Path
from yeto_cot_processor import YetoCoTProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob-path", required=True, 
                       help="Path to encrypted blobs (file or directory)")
    parser.add_argument("--provider", default="codex",
                       choices=["codex", "openai", "claude"],
                       help="API provider (codex=local codex_client, default)")
    parser.add_argument("--model", default="qwen35-9b",
                       help="Yeto model alias or HF ID")
    parser.add_argument("--output", default="yeto_output",
                       help="Output directory for training")
    parser.add_argument("--budget", type=float, default=40,
                       help="Budget in $/hour for training")
    parser.add_argument("--gpu", help="GPU specification for Yeto")
    parser.add_argument("--skip-yeto", action="store_true",
                       help="Only extract CoT, don't run Yeto")
    args = parser.parse_args()
    
    # Step 1: Extract CoT using local credentials
    print("=== Extracting CoT from encrypted blobs ===")
    print(f"Using provider: {args.provider}")
    
    processor = YetoCoTProcessor(
        provider=args.provider,
        output_dir=args.output
    )
    dataset_path = processor.process_blobs(args.blob_path)
    
    if args.skip_yeto:
        print(f"\n✅ Extraction complete! Dataset saved to {dataset_path}")
        return
    
    # Step 2: Verify dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at {dataset_path}")
        sys.exit(1)
        
    # Step 3: Build Yeto command
    print("\n=== Launching Yeto fine-tuning ===")
    yeto_cmd = [
        "yeto", "launch",
        "--model", args.model,
        "--data", dataset_path,
        "--output", f"{args.output}/model",
        "--budget", str(args.budget)
    ]
    
    if args.gpu:
        yeto_cmd.extend(["--gpu", args.gpu])
    
    print(f"Running: {' '.join(yeto_cmd)}")
    
    # Step 4: Launch Yeto
    try:
        subprocess.run(yeto_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Yeto failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 'yeto' command not found. Make sure Yeto is installed.")
        print("   Run: pip install -e '.[launcher]'")
        sys.exit(1)
        
    print(f"\n✅ Complete! Model saved to {args.output}/model")

if __name__ == "__main__":
    main()