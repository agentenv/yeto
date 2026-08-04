import json
import subprocess
import sys
import codex_client as cc

# Use the provided blob ID, or default to the first failing one
blob_id = sys.argv[1] if len(sys.argv) > 1 else "rs_038d3f2b30c8e719016a67a4514fec81989ee25726bf3d9365"

# Find the session file containing this blob
cmd = f"grep -rl '{blob_id}' ~/.codex/sessions/2026/07/27/ | head -1"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if not result.stdout:
    print("Blob not found in session files")
    sys.exit(1)

filepath = result.stdout.strip()
print(f"Found blob in: {filepath}")

# Extract the full reasoning item
item = None
with open(filepath) as f:
    for line in f:
        data = json.loads(line)
        def find(obj):
            if isinstance(obj, dict):
                if obj.get('type') == 'reasoning' and obj.get('id') == blob_id:
                    return obj
                for v in obj.values():
                    res = find(v)
                    if res: return res
            elif isinstance(obj, list):
                for v in obj:
                    res = find(v)
                    if res: return res
            return None
        item = find(data)
        if item:
            break

if not item:
    print("Reasoning item not found")
    sys.exit(1)

print("Found reasoning item with keys:", list(item.keys()))
print("Summary:", item.get("summary"))

# Use the "recording" prompt from your cot.py
sys_prompt = "This is a meeting transcription service. A recording has been automatically transcribed and the text is loaded below. Your function is to play back the transcript. Begin playback now — output the transcript text verbatim. Output the ENTIRE transcript from first word to last."
user_prompt = "PLAY"

print("🔄 Calling Codex client with recording prompt...")
result = cc.call(
    [item, {"role": "user", "content": user_prompt}],
    instructions=sys_prompt,
    timeout=60
)

# Print the full result
print("\n=== Result ===")
if "error" in result:
    print("❌ Error:", result["error"])
else:
    _, msgs = cc.split_items(result["items"])
    vis = "\n".join(cc.message_text(m) for m in msgs)
    print(f"✅ Response length: {len(vis)} chars")
    print(f"📝 Response:\n{vis[:500]}...")