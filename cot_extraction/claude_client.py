#!/usr/bin/env python3
"""Claude client using local session token from ~/.claude/auth.json."""
import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class ClaudeClient:
    def __init__(self):
        self.token = self._load_token()
        self.model = "claude-3-opus-20240229"  # adjust as needed
        self.base_url = "https://claude.ai/api"

    def _load_token(self) -> Optional[str]:
        # 1. Try file ~/.claude/auth.json
        auth_file = Path.home() / '.claude' / 'auth.json'
        if auth_file.exists():
            try:
                with open(auth_file) as f:
                    data = json.load(f)
                if 'sessionKey' in data:
                    return data['sessionKey']
            except:
                pass
        # 2. Try environment variable
        token = os.environ.get('CLAUDE_SESSION_TOKEN')
        if token:
            return token
        return None

    def call(self, items, instructions="", effort="high", model=None, timeout=300):
        if not self.token:
            return {"error": "No Claude token. Please save sessionKey to ~/.claude/auth.json or set CLAUDE_SESSION_TOKEN"}

        # Build messages
        messages = []
        for item in items:
            if item.get("type") == "reasoning":
                content = item.get("encrypted_content", "")
                messages.append({"role": "user", "content": content})
            elif "role" in item and "content" in item:
                messages.append({"role": item["role"], "content": item["content"]})
            else:
                messages.append({"role": "user", "content": json.dumps(item)})

        model = model or self.model

        # Claude backend expects messages in the same format as the web app
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if instructions:
            payload["system"] = instructions

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url}/messages"  # or /chat/conversations

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                # Extract assistant's message
                content = data.get("content", [])
                if content and content[0].get("type") == "text":
                    return {"items": [{"role": "assistant", "content": content[0]["text"]}]}
                else:
                    return {"error": "No text in response"}
            else:
                return {"error": f"API error {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

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
        _default_client = ClaudeClient()
    return _default_client

def call(*args, **kwargs):
    return get_client().call(*args, **kwargs)
def split_items(*args, **kwargs):
    return get_client().split_items(*args, **kwargs)
def message_text(*args, **kwargs):
    return get_client().message_text(*args, **kwargs)
def extract_encrypted_content(*args, **kwargs):
    return get_client().extract_encrypted_content(*args, **kwargs)