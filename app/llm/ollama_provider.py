"""Local Ollama LLM Provider implementation for AO-03.
Uses the Ollama REST API for fully local inference — no API key required.
"""

import json
import logging
import requests
from app.config import Config
from app.llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)

# Sentinel used internally; never exposed to end-users.
_OLLAMA_CONNECT_ERROR = "OLLAMA_UNREACHABLE"


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or Config.OLLAMA_MODEL or "qwen2.5:0.5b"
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self._timeout = 60  # seconds — local models can be slow on first load

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns True if Ollama is reachable and the configured model exists."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            try:
                data = resp.json()
            except Exception:
                return False
            models = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
            target = self.model.lower()
            target_base = target.split(":")[0]
            # Ollama lists models as "name:tag"; match with or without tag
            return any(
                target == m.lower()
                or target == m.lower().split(":")[0]
                or target_base == m.lower()
                or target_base == m.lower().split(":")[0]
                for m in models
            )
        except requests.ConnectionError:
            logger.warning(
                "Ollama is not reachable at %s. "
                "Please start Ollama and ensure the model '%s' is pulled.",
                self.base_url,
                self.model,
            )
            return False
        except Exception as exc:
            logger.warning("Ollama availability check failed: %s", exc)
            return False

    def generate(self, system_prompt: str, user_prompt: str, stream: bool = False) -> str:
        """Sends a chat-completion request to the local Ollama instance.

        Uses ``/api/chat`` with system + user messages so that the model
        properly separates instruction context from the investigation data.
        Sets ``"format": "json"`` to enforce JSON output.
        Handles both single-payload JSON and streaming newline-delimited JSON (NDJSON).
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "format": "json",
            "options": {
                "temperature": 0.2,
            },
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, application/x-ndjson",
                },
                json=payload,
                timeout=self._timeout,
            )
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Please ensure Ollama is running "
                f"at {self.base_url} and the model '{self.model}' is available. "
                "Start Ollama with: ollama serve"
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self._timeout}s. "
                "The model may still be loading. Please try again."
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama network request failed: {exc}")

        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{self.model}' not found on Ollama. "
                f"Pull it first with: ollama pull {self.model}"
            )

        if response.status_code != 200:
            logger.error(
                "Ollama returned HTTP %s: %s",
                response.status_code,
                response.text[:500],
            )
            raise RuntimeError(
                f"Ollama returned an error (HTTP {response.status_code}). "
                "Check the application logs for details."
            )

        raw_text = response.text
        if not raw_text or not raw_text.strip():
            logger.error("Ollama returned an empty response body (HTTP %s)", response.status_code)
            raise RuntimeError("Ollama returned an empty response body.")

        # Attempt to parse as single JSON document first
        res_data = None
        try:
            res_data = response.json()
        except (ValueError, json.JSONDecodeError):
            res_data = None

        content = None

        # 1. Single JSON object response
        if isinstance(res_data, dict):
            if "message" in res_data and isinstance(res_data["message"], dict):
                content = res_data["message"].get("content", "")
            elif "response" in res_data:
                content = res_data.get("response", "")
            elif "error" in res_data:
                raise RuntimeError(f"Ollama API error: {res_data['error']}")
            else:
                logger.error(
                    "Unexpected Ollama response structure: %s",
                    json.dumps(res_data)[:500],
                )
                raise RuntimeError(
                    "Unexpected Ollama response structure. "
                    "Check the application logs for details."
                )
        else:
            # 2. Streaming NDJSON response (newline-delimited JSON chunks)
            chunks = []
            for line in raw_text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if isinstance(chunk, dict):
                        if "message" in chunk and isinstance(chunk["message"], dict):
                            chunks.append(chunk["message"].get("content", ""))
                        elif "response" in chunk:
                            chunks.append(chunk.get("response", ""))
                        elif "error" in chunk:
                            raise RuntimeError(f"Ollama API error: {chunk['error']}")
                except (ValueError, json.JSONDecodeError):
                    pass

            if chunks:
                content = "".join(chunks)
            else:
                logger.error("Ollama returned non-JSON response: %s", raw_text[:500])
                raise RuntimeError(
                    "Ollama returned a malformed response. "
                    "Check the application logs for details."
                )

        if content is None or not str(content).strip():
            logger.error("Ollama response contained no generated text.")
            raise RuntimeError("Ollama response contained no generated text.")

        return str(content).strip()
