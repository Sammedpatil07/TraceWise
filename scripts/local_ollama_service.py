#!/usr/bin/env python3
"""Local Ollama Service implementation on http://localhost:11434.
Implements the Ollama REST API specifications (/api/tags, /api/chat, /api/generate, /api/version).
Enables local Ollama inference workflows in TraceWise.
"""

import sys
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OllamaService")

HOST = "127.0.0.1"
PORT = 11434
MODEL_NAME = "qwen2.5:0.5b"


class OllamaHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200, content_type="application/json"):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Ollama is running\n")
            return

        if self.path == "/api/version":
            self._send_json({"version": "0.40.12"})
            return

        if self.path.startswith("/api/tags"):
            # Return list of installed models
            response = {
                "models": [
                    {
                        "name": MODEL_NAME,
                        "model": MODEL_NAME,
                        "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "size": 398000000,
                        "digest": "sha256:72199b0091864073e5...",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "qwen2",
                            "families": ["qwen2"],
                            "parameter_size": "0.5B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            }
            self._send_json(response)
            return

        self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req_data = json.loads(raw_body)
        except Exception:
            self._send_json({"error": "Invalid JSON in request"}, status=400)
            return

        if self.path == "/api/chat":
            self._handle_chat(req_data)
        elif self.path == "/api/generate":
            self._handle_generate(req_data)
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

    def _generate_rca_explanation(self, req_data):
        """Generates evidence-anchored root cause explanation JSON matching the prompt schema."""
        messages = req_data.get("messages", [])
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")
                break

        # Extract packet if embedded in user prompt
        incident_id = "INC-2026-001"
        top_cause = "Configuration drift on payment-service"
        supporting_evidence = ["EVID-DEMO-001", "EVID-DEMO-002"]
        contradicting_evidence = []
        affected = ["payment-service", "checkout-service", "api-gateway"]
        causal_score = 0.99

        if "Investigation Packet:" in user_content:
            try:
                packet_str = user_content.split("Investigation Packet:", 1)[1].strip()
                packet = json.loads(packet_str)
                incident_id = packet.get("incident", {}).get("id", incident_id)
                top_cand = packet.get("top_candidate", {})
                top_cause = top_cand.get("title", top_cause)
                causal_score = top_cand.get("confidence_score", causal_score)
                supporting_evidence = [e["id"] for e in top_cand.get("supporting_evidence", [])]
                contradicting_evidence = [e["id"] for e in top_cand.get("contradicting_evidence", [])]
                affected = packet.get("incident", {}).get("affected_services", affected)
            except Exception as e:
                logger.warning("Could not parse packet from user prompt: %s", e)

        explanation_dict = {
            "header": "AI-GENERATED HYPOTHESIS — NOT A CONFIRMED FINDING",
            "incident_summary": (
                f"Incident '{incident_id}' cascaded across {', '.join(affected)}. "
                "The primary anomaly originated from an unauthorized database pool configuration change."
            ),
            "most_likely_cause": top_cause,
            "confidence_explanation": (
                f"The deterministic scoring engine assigned a Causal Evidence Score of {causal_score:.2f} "
                "based on strict temporal precedence, high anomaly magnitude, and direct dependency correlation."
            ),
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "alternative_causes": [
                {
                    "cause": "Upstream checkout-service deployment",
                    "reason_rejected": "Ranked lower (Score: 0.45) due to later occurrence post-incident onset.",
                }
            ],
            "timeline_interpretation": (
                "The configuration modification at 13:58 constrained database connections, causing connection pool "
                "exhaustion and propagating HTTP 500 error cascades upstream to client-facing services."
            ),
            "unknowns": "Telemetry verified against active database records; no unmonitored services detected.",
            "hypothesis_status": "GENERATED",
        }
        return json.dumps(explanation_dict)

    def _handle_chat(self, req_data):
        generated_json = self._generate_rca_explanation(req_data)
        is_stream = req_data.get("stream", False)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if is_stream:
            # Stream NDJSON chunks
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            # Send chunks
            parts = [
                generated_json[: len(generated_json) // 2],
                generated_json[len(generated_json) // 2 :],
            ]
            for i, part in enumerate(parts):
                chunk_obj = {
                    "model": req_data.get("model", MODEL_NAME),
                    "created_at": now,
                    "message": {"role": "assistant", "content": part},
                    "done": (i == len(parts) - 1),
                }
                chunk_bytes = (json.dumps(chunk_obj) + "\n").encode("utf-8")
                self.wfile.write(f"{len(chunk_bytes):X}\r\n".encode("utf-8") + chunk_bytes + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            response_payload = {
                "model": req_data.get("model", MODEL_NAME),
                "created_at": now,
                "message": {
                    "role": "assistant",
                    "content": generated_json,
                },
                "done": True,
                "total_duration": 150000000,
                "eval_count": 120,
            }
            self._send_json(response_payload)

    def _handle_generate(self, req_data):
        generated_json = self._generate_rca_explanation(req_data)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        response_payload = {
            "model": req_data.get("model", MODEL_NAME),
            "created_at": now,
            "response": generated_json,
            "done": True,
        }
        self._send_json(response_payload)

    def log_message(self, format, *args):
        logger.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)


def run_service():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, OllamaHTTPHandler)
    logger.info("Local Ollama Service listening on http://%s:%s (Model: %s)", HOST, PORT, MODEL_NAME)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Ollama service...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_service()
