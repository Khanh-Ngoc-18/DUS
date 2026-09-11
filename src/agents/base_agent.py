import yaml
import os
import time
import requests


def load_model_config(path: str = "config/model_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseAgent:
    def __init__(self, role: str, config_path: str = "config/model_config.yaml"):
        self.role = role
        cfg = load_model_config(config_path)

        model_cfg = cfg["model"]
        agent_cfg = cfg["agents"].get(role, {})

        # Mỗi agent có provider/name riêng, fallback về model_cfg
        self.provider = agent_cfg.get("provider", model_cfg.get("provider"))
        self.model    = agent_cfg.get("name",     model_cfg.get("name"))

        self.temperature = agent_cfg.get("temperature", model_cfg["temperature"])
        self.max_tokens  = agent_cfg.get("max_tokens",  model_cfg["max_tokens"])
        self.top_p       = model_cfg.get("top_p", 1.0)

        self.total_tokens_used = 0

    def call_llm(self, prompt: str) -> tuple[str, int]:
        if self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt)
        elif self.provider == "chatgpt":
            return self._call_openai(prompt)
        elif self.provider == "openrouter":
            return self._call_openrouter(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_ollama(self, prompt: str, max_retries: int = 3) -> tuple[str, int]:
            last_err = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "num_predict": self.max_tokens,
                                # Seed lay tu MAD_SEED (do MAD.py dat theo --seed). Cung 1 seed
                                # => tai lap duoc; seed khac nhau => rollout doc lap (multi-seed).
                                "seed": int(os.environ.get("MAD_SEED", "0")),
                            },
                        },
                        timeout=(10, 300),  # 10s connect, 300s đợi generate
                    )
                    response.raise_for_status()
                    data = response.json()
                    if "response" not in data:
                        raise RuntimeError(f"Ollama error: {data.get('error', data)}")
                    text = data["response"]
                    if os.environ.get("MAD_DEBUG"):
                        print(f"[RAW {self.role}] {text[:800]!r}")
                    tokens = data.get("eval_count", len(text.split()))
                    self.total_tokens_used += tokens
                    return text, tokens

                except (requests.Timeout, requests.ConnectionError,
                        requests.HTTPError, RuntimeError) as e:
                    last_err = e
                    wait = 5 * (attempt + 1)
                    print(f"[Ollama ERROR] {e} → retry in {wait}s ({attempt+1}/{max_retries})")
                    time.sleep(wait)

            raise RuntimeError(f"Ollama failed after {max_retries} retries: {last_err}")

    def _call_groq(self, prompt: str, max_retries: int = 5) -> tuple[str, int]:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")

        client = Groq(api_key=api_key)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content
                usage = response.usage
                tokens = usage.total_tokens if usage else len(text.split())
                self.total_tokens_used += tokens
                return text, tokens

            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"[Groq ERROR] {e} → retry in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)

        raise RuntimeError("Groq failed after retries")

    def _call_gemini(self, prompt: str, max_retries: int = 5) -> tuple[str, int]:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")

        client = genai.Client(api_key=api_key)

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        top_p=self.top_p,
                    ),
                )
                text = response.text
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    tokens = (getattr(usage, "prompt_token_count", 0) or 0) + \
                             (getattr(usage, "candidates_token_count", 0) or 0)
                else:
                    tokens = len(text.split())
                self.total_tokens_used += tokens
                return text, tokens

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "resource_exhausted" in err or "quota" in err:
                    wait = 15 * (attempt + 1)
                    print(f"Gemini rate limit, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Gemini rate limit exceeded after {max_retries} retries.")

    def _call_deepseek(self, prompt: str, max_retries: int = 5) -> tuple[str, int]:
        from openai import OpenAI, RateLimitError

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set.")

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content
                usage = response.usage
                tokens = usage.total_tokens if usage else len(text.split())
                self.total_tokens_used += tokens
                return text, tokens

            except RateLimitError:
                wait = 15 * (attempt + 1)
                print(f"DeepSeek rate limit, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)

        raise RuntimeError(f"DeepSeek rate limit exceeded after {max_retries} retries.")

    def _call_openai(self, prompt: str, max_retries: int = 5) -> tuple[str, int]:
        from openai import OpenAI
        import random

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")

        client = OpenAI(api_key=api_key)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content
                usage = response.usage
                tokens = usage.total_tokens if usage else len(text.split())
                self.total_tokens_used += tokens
                return text, tokens

            except Exception as e:
                wait = (10 * (attempt + 1)) + random.uniform(0, 3)
                print(f"[OpenAI ERROR] {e} → retry in {wait:.1f}s ({attempt+1}/{max_retries})")
                time.sleep(wait)

        raise RuntimeError("OpenAI failed after retries")

    def _call_openrouter(self, prompt: str, max_retries: int = 5) -> tuple[str, int]:
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set.")

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content
                usage = response.usage
                tokens = usage.total_tokens if usage else len(text.split())
                self.total_tokens_used += tokens
                return text, tokens

            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"[OpenRouter ERROR] {e} → retry in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)

        raise RuntimeError("OpenRouter failed after retries")