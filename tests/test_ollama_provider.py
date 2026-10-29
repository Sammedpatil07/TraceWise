"""Unit and regression tests for the Ollama LLM Provider and Explanation Service.
All tests mock HTTP responses — no running Ollama instance required.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from app.llm.ollama_provider import OllamaProvider
from app.llm.explanation import ExplanationService
from app.models.incident import Incident, CandidateCause, TimelineEvent
from app.models.evidence import NormalizedEvidence, EvidenceType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tags_response(models=None, status_code: int = 200):
    """Return a mock /api/tags response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps({
        "models": [{"name": m} for m in (models or ["qwen2.5:0.5b"])]
    })
    resp.json.return_value = {
        "models": [{"name": m} for m in (models or ["qwen2.5:0.5b"])]
    }
    return resp


def _mock_chat_response(content: str, status_code: int = 200, as_json: bool = True):
    """Return a mock /api/chat response."""
    resp = MagicMock()
    resp.status_code = status_code
    if as_json:
        payload = {"message": {"role": "assistant", "content": content}}
        resp.text = json.dumps(payload)
        resp.json.return_value = payload
    else:
        resp.text = content
        resp.json.side_effect = ValueError("Not JSON")
    return resp


def _mock_streaming_chat_response(chunks, status_code: int = 200):
    """Return a mock streaming NDJSON /api/chat response."""
    resp = MagicMock()
    resp.status_code = status_code
    lines = [
        json.dumps({"message": {"role": "assistant", "content": c}, "done": (i == len(chunks) - 1)})
        for i, c in enumerate(chunks)
    ]
    ndjson_body = "\n".join(lines)
    resp.text = ndjson_body
    # NDJSON cannot be parsed as a single JSON document by response.json()
    resp.json.side_effect = ValueError("Extra data / NDJSON")
    resp.headers = {"Content-Type": "application/x-ndjson"}
    return resp


def _create_sample_incident_data():
    """Builds sample incident, candidates, timeline, and evidence for ExplanationService."""
    incident = Incident(
        id="INC-2026-001",
        title="Checkout Latency Spike and Error Surge",
        description="High HTTP 500 error rates on payment-service cascaded to checkout.",
        started_at="2026-10-28T14:00:00Z",
        resolved_at="2026-10-28T14:45:00Z",
        affected_services=["payment-service", "checkout-service"],
    )

    top_candidate = CandidateCause(
        id="cand_payment_config",
        incident_id="INC-2026-001",
        service_id="payment-service",
        title="Configuration drift on payment-service",
        description="Database connection pool size was erroneously reduced to 2.",
        candidate_type="CONFIGURATION_DRIFT",
        rank=1,
        evidence_score=0.95,
        confidence_score=0.95,
        scoring_breakdown={
            "temporal_precedence": 1.0,
            "anomaly_magnitude": 0.9,
            "error_correlation": 0.95,
            "change_proximity": 1.0,
        },
        supporting_evidence_ids=["EVID-DEMO-001", "EVID-DEMO-002"],
        contradicting_evidence_ids=[],
    )

    ranked_candidates = [
        top_candidate,
        CandidateCause(
            id="cand_checkout_deploy",
            incident_id="INC-2026-001",
            service_id="checkout-service",
            title="Deploy on checkout-service",
            description="Upstream checkout deployment.",
            candidate_type="DEPLOYMENT",
            rank=2,
            evidence_score=0.45,
            confidence_score=0.45,
            scoring_breakdown={},
            supporting_evidence_ids=[],
            contradicting_evidence_ids=[],
        ),
    ]

    timeline = [
        TimelineEvent(
            id="tl_001",
            incident_id="INC-2026-001",
            timestamp="2026-10-28T13:58:00Z",
            event_type="CONFIG",
            service_id="payment-service",
            summary="DB pool size changed from 50 to 2",
            evidence_id="EVID-DEMO-001",
        ),
        TimelineEvent(
            id="tl_002",
            incident_id="INC-2026-001",
            timestamp="2026-10-28T14:01:00Z",
            event_type="ERROR",
            service_id="payment-service",
            summary="HTTP 500 error surge observed",
            evidence_id="EVID-DEMO-002",
        ),
    ]

    evidence_list = [
        NormalizedEvidence(
            id="EVID-DEMO-001",
            incident_id="INC-2026-001",
            service_id="payment-service",
            evidence_type=EvidenceType.CONFIGURATION,
            summary="DB pool size changed from 50 to 2",
            timestamp="2026-10-28T13:58:00Z",
        ),
        NormalizedEvidence(
            id="EVID-DEMO-002",
            incident_id="INC-2026-001",
            service_id="payment-service",
            evidence_type=EvidenceType.LOG,
            summary="HTTP 500 error surge on payment-service",
            timestamp="2026-10-28T14:01:00Z",
        ),
    ]

    return incident, top_candidate, ranked_candidates, timeline, evidence_list


# ---------------------------------------------------------------------------
# is_available Tests
# ---------------------------------------------------------------------------

class TestOllamaIsAvailable:
    @patch("app.llm.ollama_provider.requests.get")
    def test_available_when_model_exists(self, mock_get):
        mock_get.return_value = _mock_tags_response(["qwen2.5:0.5b"])
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        assert provider.is_available() is True

    @patch("app.llm.ollama_provider.requests.get")
    def test_available_when_model_matches_base_name(self, mock_get):
        mock_get.return_value = _mock_tags_response(["qwen2.5:latest"])
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5")
        assert provider.is_available() is True

    @patch("app.llm.ollama_provider.requests.get")
    def test_unavailable_when_model_missing(self, mock_get):
        mock_get.return_value = _mock_tags_response(["llama3:latest"])
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        assert provider.is_available() is False

    @patch("app.llm.ollama_provider.requests.get")
    def test_unavailable_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        assert provider.is_available() is False

    @patch("app.llm.ollama_provider.requests.get")
    def test_unavailable_on_http_error(self, mock_get):
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        assert provider.is_available() is False

    @patch("app.llm.ollama_provider.requests.get")
    def test_unavailable_on_malformed_json(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        assert provider.is_available() is False


# ---------------------------------------------------------------------------
# generate — Successful Generation & Regression Tests
# ---------------------------------------------------------------------------

class TestOllamaGenerate:
    @patch("app.llm.ollama_provider.requests.post")
    def test_successful_generation(self, mock_post):
        """Verifies standard valid Ollama response returns non-empty text."""
        expected = json.dumps({"header": "AI-GENERATED", "summary": "test explanation"})
        mock_post.return_value = _mock_chat_response(expected)
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        result = provider.generate("system prompt", "user prompt")
        assert result == expected

    @patch("app.llm.ollama_provider.requests.post")
    def test_payload_specifies_format_json_and_non_streaming(self, mock_post):
        """Verifies that format: json and stream: false are explicitly sent in payload."""
        mock_post.return_value = _mock_chat_response('{"result": "ok"}')
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        provider.generate("sys", "usr")

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["format"] == "json"
        assert payload["stream"] is False
        assert payload["model"] == "qwen2.5:0.5b"

    @patch("app.llm.ollama_provider.requests.post")
    def test_streaming_ndjson_response(self, mock_post):
        """Verifies streaming NDJSON chunks are seamlessly parsed and accumulated into full text."""
        chunks = [
            '{"header": "AI-GENERATED", ',
            '"incident_summary": "DB pool drained", ',
            '"most_likely_cause": "Configuration drift"}',
        ]
        mock_post.return_value = _mock_streaming_chat_response(chunks)
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        result = provider.generate("sys", "usr", stream=True)

        expected = "".join(chunks)
        assert result == expected

    @patch("app.llm.ollama_provider.requests.post")
    def test_api_generate_response_format(self, mock_post):
        """Verifies that Ollama /api/generate format ('response' key) is supported."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = json.dumps({"response": '{"summary": "generated via generate API"}'})
        resp.json.return_value = {"response": '{"summary": "generated via generate API"}'}
        mock_post.return_value = resp

        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        result = provider.generate("sys", "usr")
        assert result == '{"summary": "generated via generate API"}'


# ---------------------------------------------------------------------------
# generate — Error Handling & Edge Cases
# ---------------------------------------------------------------------------

class TestOllamaErrors:
    @patch("app.llm.ollama_provider.requests.post")
    def test_connection_error(self, mock_post):
        """Verifies clear diagnostic when connection is refused."""
        mock_post.side_effect = requests.ConnectionError("refused")
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_timeout_error(self, mock_post):
        """Verifies clear diagnostic on request timeout."""
        mock_post.side_effect = requests.Timeout("timed out")
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_model_not_found_404(self, mock_post):
        """Verifies clear diagnostic on HTTP 404 model not found."""
        resp = MagicMock()
        resp.status_code = 404
        mock_post.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="nonexistent:latest")
        with pytest.raises(RuntimeError, match="not found on Ollama"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_server_error_500(self, mock_post):
        """Verifies clear diagnostic on HTTP 500 error."""
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "internal server error"
        mock_post.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_empty_response_body(self, mock_post):
        """Verifies clear diagnostic when Ollama returns HTTP 200 with empty body."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        mock_post.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="empty response body"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_missing_generated_text(self, mock_post):
        """Verifies clear diagnostic when response JSON has empty content string."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = json.dumps({"message": {"role": "assistant", "content": "   "}})
        resp.json.return_value = {"message": {"role": "assistant", "content": "   "}}
        mock_post.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="no generated text"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_malformed_json_response(self, mock_post):
        """Verifies clear diagnostic when response is not valid JSON."""
        mock_post.return_value = _mock_chat_response("not json at all", as_json=False)
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="malformed response"):
            provider.generate("sys", "usr")

    @patch("app.llm.ollama_provider.requests.post")
    def test_unexpected_response_structure(self, mock_post):
        """Verifies clear diagnostic when JSON lacks expected message or response fields."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = json.dumps({"unexpected_key": "value"})
        resp.json.return_value = {"unexpected_key": "value"}
        mock_post.return_value = resp
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:0.5b")
        with pytest.raises(RuntimeError, match="Unexpected Ollama response structure"):
            provider.generate("sys", "usr")


# ---------------------------------------------------------------------------
# ExplanationService Integration & Robust Parsing Regression Tests
# ---------------------------------------------------------------------------

class TestExplanationServiceIntegration:
    def setup_method(self):
        self.service = ExplanationService()
        self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence = (
            _create_sample_incident_data()
        )

    @patch.object(OllamaProvider, "is_available", return_value=True)
    @patch.object(OllamaProvider, "generate")
    def test_valid_ollama_response_used_by_explanation(self, mock_gen, mock_avail):
        """Acceptance test: valid Ollama output is parsed and returned with status AVAILABLE."""
        expected_hyp = {
            "header": "AI-GENERATED HYPOTHESIS — NOT A CONFIRMED FINDING",
            "incident_summary": "Database pool exhaustion triggered cascaded failure.",
            "most_likely_cause": "Configuration drift on payment-service",
            "confidence_explanation": "Causal evidence score 0.95 with temporal precedence.",
            "supporting_evidence": ["EVID-DEMO-001", "EVID-DEMO-002"],
            "contradicting_evidence": [],
            "alternative_causes": [{"cause": "Deploy", "reason_rejected": "Occurred later"}],
            "timeline_interpretation": "Config changed at 13:58 followed by 500 error surge.",
            "unknowns": "No deep locking telemetry.",
            "hypothesis_status": "GENERATED",
        }
        mock_gen.return_value = json.dumps(expected_hyp)

        explanation, status = self.service.generate_explanation(
            self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence
        )

        assert status == "AVAILABLE"
        assert explanation["hypothesis_status"] == "AI_GENERATED"
        assert explanation["most_likely_cause"] == "Configuration drift on payment-service"
        assert "EVID-DEMO-001" in explanation["supporting_evidence"]

    @patch.object(OllamaProvider, "is_available", return_value=True)
    @patch.object(OllamaProvider, "generate")
    def test_markdown_code_fenced_response_parsed_cleanly(self, mock_gen, mock_avail):
        """Regression test for terminal error: ```json ... ``` markdown does NOT trigger JSONDecodeError."""
        inner_json = {
            "incident_summary": "Checkout failure caused by database pool config.",
            "most_likely_cause": "Configuration drift on payment-service",
            "supporting_evidence": ["EVID-DEMO-001"],
            "contradicting_evidence": [],
        }
        # Simulate LLM wrapping response in markdown code blocks
        mock_gen.return_value = f"```json\n{json.dumps(inner_json, indent=2)}\n```"

        explanation, status = self.service.generate_explanation(
            self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence
        )

        assert status == "AVAILABLE"
        assert explanation["hypothesis_status"] == "AI_GENERATED"
        assert explanation["most_likely_cause"] == "Configuration drift on payment-service"

    @patch.object(OllamaProvider, "is_available", return_value=True)
    @patch.object(OllamaProvider, "generate")
    def test_preamble_surrounding_json_extracted_cleanly(self, mock_gen, mock_avail):
        """Verifies text with conversational intro and outro is correctly parsed."""
        inner_json = {
            "incident_summary": "Preamble test summary",
            "most_likely_cause": "Configuration drift on payment-service",
        }
        mock_gen.return_value = (
            f"Here is the investigation analysis:\n{json.dumps(inner_json)}\nI hope this helps!"
        )

        explanation, status = self.service.generate_explanation(
            self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence
        )

        assert status == "AVAILABLE"
        assert explanation["most_likely_cause"] == "Configuration drift on payment-service"

    @patch.object(OllamaProvider, "is_available", return_value=True)
    @patch.object(OllamaProvider, "generate")
    def test_fallback_on_malformed_llm_response(self, mock_gen, mock_avail):
        """Verifies graceful fallback to deterministic heuristic when Ollama returns malformed text."""
        mock_gen.return_value = "This is plain conversational text without any JSON object."

        explanation, status = self.service.generate_explanation(
            self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence
        )

        assert status == "FALLBACK_HEURISTIC"
        assert explanation["hypothesis_status"] == "DETERMINISTIC_HEURISTIC"
        assert explanation["most_likely_cause"] == self.top_cand.title

    @patch.object(OllamaProvider, "is_available", return_value=True)
    @patch.object(OllamaProvider, "generate")
    def test_fallback_on_generation_exception(self, mock_gen, mock_avail):
        """Verifies graceful fallback when generate() raises RuntimeError."""
        mock_gen.side_effect = RuntimeError("Ollama connection timed out")

        explanation, status = self.service.generate_explanation(
            self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence
        )

        assert status == "FALLBACK_HEURISTIC"
        assert explanation["hypothesis_status"] == "DETERMINISTIC_HEURISTIC"

    @patch.object(OllamaProvider, "is_available", return_value=False)
    def test_fallback_when_ollama_unavailable(self, mock_avail):
        """Verifies clear labeling as DETERMINISTIC_HEURISTIC when Ollama service is offline."""
        explanation, status = self.service.generate_explanation(
            self.incident, self.top_cand, self.ranked_cands, self.timeline, self.evidence
        )

        assert status == "FALLBACK_HEURISTIC"
        assert explanation["hypothesis_status"] == "DETERMINISTIC_HEURISTIC"
        assert "EVID-DEMO-001" in explanation["supporting_evidence"]
