"""The three outside pieces: local SLM, Gemini, and Serper web search.

Each has one method, so tests (and other back-ends) can swap in fakes.
"""
import os
import re
import time

import requests


class LocalSLM:
    """A ~3B instruction-tuned model running locally (4-bit on a GPU)."""

    def __init__(self, name="meta-llama/Llama-3.2-3B-Instruct", load_in_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # float16, not bfloat16: the free Kaggle/Colab T4 GPU has no native bfloat16.
        quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                 if load_in_4bit else None)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", quantization_config=quant, torch_dtype=torch.float16)
        self.name = name

    def generate(self, prompt, max_new_tokens=192):
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        # Decode only the new tokens. (The original code decoded the prompt too,
        # so its parser read the template line "WHY: <why_question>" as the query.)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


class Gemini:
    """The large model, used only for the final answer. Counts tokens, retries on rate limits."""

    def __init__(self, name="gemini-2.5-flash-lite", api_key=None, pause=0.5):
        from google import genai

        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.name, self.pause = name, pause
        self.tokens = {"prompt": 0, "output": 0, "calls": 0}

    def generate(self, prompt, retries=4):
        for attempt in range(retries):
            try:
                resp = self.client.models.generate_content(model=self.name, contents=prompt)
                u = resp.usage_metadata
                if u is not None:
                    self.tokens["prompt"] += u.prompt_token_count or 0
                    self.tokens["output"] += u.candidates_token_count or 0
                self.tokens["calls"] += 1
                time.sleep(self.pause)
                return resp.text or ""
            except Exception as e:  # rate limits: wait as long as the API asks
                m = re.search(r"retry in ([0-9.]+)s", str(e))
                wait = float(m.group(1)) if m else 30.0 * (attempt + 1)
                print(f"[gemini] {type(e).__name__}; retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("Gemini failed after retries")


class SerperSearch:
    """Google results through the Serper API. Returns [{'snippet', 'url'}]."""

    URL = "https://google.serper.dev/search"

    def __init__(self, api_key=None, timeout=10):
        self.key = api_key or os.environ["SERPER_API_KEY"]
        self.timeout = timeout

    def search(self, query, k=2):
        r = requests.post(self.URL, json={"q": query}, timeout=self.timeout,
                          headers={"X-API-KEY": self.key, "Content-Type": "application/json"})
        r.raise_for_status()
        out = []
        for item in r.json().get("organic", []):
            snippet = (item.get("snippet") or "").replace("\n", " ").strip()
            if snippet:
                out.append({"snippet": snippet, "url": item.get("link", "")})
            if len(out) == k:
                break
        return out
