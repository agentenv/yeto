# cot_extraction/yeto_cot_processor.py
"""Process extracted CoT for Yeto training."""
import json
import os
from pathlib import Path
from typing import List, Dict
from cot_extractor import CoTExtractor
import glob

class YetoCoTProcessor:
    def __init__(self, provider: str = "codex", output_dir: str = "cot_training"):
        """
        Initialize processor.
        provider: "codex", "openai", or "claude" - defaults to codex_client
        """
        self.extractor = CoTExtractor(provider)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def process_blobs(self, blob_path: str) -> str:
        """Process encrypted blobs and create Yeto training data."""
        # Load blobs
        blobs = self._load_blobs(blob_path)
        print(f"Found {len(blobs)} blobs")
        
        if not blobs:
            print("No blobs found to process")
            return str(self.output_dir / "empty.jsonl")
        
        # Extract CoT and create training examples
        training_data = []
        for i, blob in enumerate(blobs):
            print(f"Processing blob {i+1}/{len(blobs)}: {blob['id'][:30]}...")
            method, text = self.extractor.extract_one(blob['blob'], blob['id'])
            
            if text and len(text) >= 80:
                training_data.append({
                    "messages": [
                        {"role": "user", "content": "Extract the complete reasoning and transcription."},
                        {"role": "assistant", "content": text}
                    ],
                    "metadata": {
                        "blob_id": blob['id'],
                        "method": method,
                        "source": blob.get('src', 'unknown')
                    }
                })
                print(f"  ✅ Extracted {len(text)} chars using {method}")
            else:
                print(f"  ❌ Failed to extract from {blob['id'][:30]}")
                
        # Save for Yeto
        output_file = self.output_dir / "cot_training.jsonl"
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
                
        print(f"\nCreated {len(training_data)} training examples at {output_file}")
        return str(output_file)
        
    def _load_blobs(self, path: str) -> List[Dict]:
        """Load encrypted blobs from JSONL files (copied from your original code)."""
        blobs = []
        seen = set()
        
        path = Path(path)
        if path.is_file():
            jsonl_files = [path]
        else:
            jsonl_files = list(path.glob("**/*.jsonl")) + list(path.glob("**/*.json"))
            
        for file_path in jsonl_files:
            print(f"Scanning: {file_path}")
            try:
                with open(file_path) as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            self._extract_blobs_recursive(data, blobs, seen, file_path)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
        return blobs
        
    def _extract_blobs_recursive(self, obj, blobs, seen, file_path):
        """Recursively extract encrypted_content from nested objects."""
        if isinstance(obj, dict):
            # Check for encrypted_content
            ec = obj.get("encrypted_content")
            rid = obj.get("id", f"blob_{len(blobs)}")
            
            if ec and rid not in seen:
                seen.add(rid)
                blobs.append({
                    "id": rid,
                    "blob": ec,
                    "src": file_path.name[:40]
                })
                
            # Recurse into values
            for value in obj.values():
                self._extract_blobs_recursive(value, blobs, seen, file_path)
                
        elif isinstance(obj, list):
            for item in obj:
                self._extract_blobs_recursive(item, blobs, seen, file_path)