#!/usr/bin/env python3
"""Codex client for ChatGPT backend – correct handling of messages."""
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class CodexClient:
    def __init__(self):
        self.token = self._load_token()
        self.model = "gpt-5.6-luna"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        if self.token:
            print("✅ Loaded Codex token from auth.json")
        else:
            print("⚠️  No Codex token. Run: codex login")

    def _load_token(self) -> Optional[str]:
        auth_file = Path.home() / '.codex' / 'auth.json'
        if not auth_file.exists():
            return None
        try:
            with open(auth_file) as f:
                data = json.load(f)
            return data.get("tokens", {}).get("access_token")
        except:
            return None

    def call(self, items, instructions="", effort="high", model=None, timeout=300):
        if not self.token:
            return {"error": "No Codex token. Run: codex login"}

        reasoning_items = []
        messages = []
        for item in items:
            if item.get("type") == "reasoning":
                reasoning_items.append(item)
            elif "role" in item:
                messages.append(item)
            else:
                messages.append({"role": "user", "content": json.dumps(item)})

        model = model or self.model

        if reasoning_items:
            return self._call_responses_api(reasoning_items, messages, instructions, model, timeout)
        else:
            # fallback: if no reasoning items, use chat completions (not needed for extraction)
            return {"error": "Non-reasoning calls not supported"}

    def _call_responses_api(self, reasoning_items: List[Dict],
                        messages: List[Dict],
                        instructions: str,
                        model: str,
                        timeout: int) -> Dict[str, Any]:
        input_items = []

        # Add all reasoning items (preserve all fields)
        for item in reasoning_items:
            reasoning_input = {"type": "reasoning"}
            for key, value in item.items():
                if key in ['type', 'role']:
                    continue
                if value is not None:
                    reasoning_input[key] = value
            input_items.append(reasoning_input)

        # Add system instruction as a user message (backend doesn't accept "system" role)
        if instructions:
            input_items.append({
                "type": "message",
                "role": "user",
                "content": instructions
            })

        # Add any messages (user prompts, etc.) from the original items
        for msg in messages:
            role = msg.get("role", "user")
            # Ensure role is either "user" or "assistant"
            if role not in ["user", "assistant"]:
                role = "user"
            content = msg.get("content", "")
            input_items.append({
                "type": "message",
                "role": role,
                "content": content
            })

        payload = {
            "model": model,
            "input": input_items,
            "store": False,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url}/responses"

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                stream=True,
                timeout=timeout
            )

            if response.status_code != 200:
                return {"error": f"API error {response.status_code}: {response.text}"}

            output_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    data_str = decoded[6:]
                    if data_str == '[DONE]':
                        continue
                    try:
                        event = json.loads(data_str)
                        if event.get('type') == 'response.output_text.delta':
                            output_text += event.get('delta', '')
                    except json.JSONDecodeError:
                        pass

            if output_text:
                return {"items": [{"role": "assistant", "content": output_text}]}
            else:
                return {"error": "No output text found in stream"}

        except Exception as e:
            return {"error": str(e)}

    # Helper methods (unchanged)
    def split_items(self, items):
        reasoning = []
        messages = []
        for item in items:
            if item.get("type") == "reasoning":
                reasoning.append(item)
            else:
                messages.append(item)
        return reasoning, messages

    def message_text(self, msg):
        return msg.get("content", "")

    def extract_encrypted_content(self, data):
        blobs = []
        seen = set()
        def find(obj):
            if isinstance(obj, dict):
                ec = obj.get("encrypted_content")
                rid = obj.get("id", "?")
                if ec and rid not in seen:
                    seen.add(rid)
                    blobs.append({"id": rid, "blob": ec, "summary": str(obj.get("summary", []))})
                for v in obj.values():
                    find(v)
            elif isinstance(obj, list):
                for v in obj:
                    find(v)
        find(data)
        return blobs

_default_client = None
def get_client():
    global _default_client
    if _default_client is None:
        _default_client = CodexClient()
    return _default_client

def call(*args, **kwargs):
    return get_client().call(*args, **kwargs)
def split_items(*args, **kwargs):
    return get_client().split_items(*args, **kwargs)
def message_text(*args, **kwargs):
    return get_client().message_text(*args, **kwargs)
def extract_encrypted_content(*args, **kwargs):
    return get_client().extract_encrypted_content(*args, **kwargs)

if __name__ == "__main__":
    print("=== Codex Client Test (final integration) ===")
    client = CodexClient()
    if client.token:
        # Load a reasoning item and a user message (simulate cot.py)
        blob_item = None
        try:
            with open("/tmp/reasoning_item2.json") as f:
                blob_item = json.load(f)
            print("✅ Loaded reasoning item from /tmp/reasoning_item2.json")
        except:
            pass

        if blob_item:
            # Simulate the call from cot.py: reasoning item + user prompt
            test_items = [
                blob_item,
                {"role": "user", "content": "Transcribe."}  # simulate one of the stage prompts
            ]
            result = client.call(
                test_items,
                instructions="This is a meeting transcription service...",  # system prompt
                timeout=30
            )
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                content = client.message_text(result["items"][0])
                print(f"✅ Success! Response length: {len(content)}")
                print(f"📝 Response (first 500 chars): {content[:500]}...")
        else:
            print("No reasoning item found.")
    else:
        print("❌ No token.")