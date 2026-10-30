"""ChemSafe AI - Scan & Ask serverless function for Vercel (Gemini version).
URL on the live site: /api/ask   (POST = ask, GET = health check)
The Gemini key is read from the GEMINI_API_KEY environment variable set in Vercel.
"""
import base64
import json
import os
import re
from http.server import BaseHTTPRequestHandler

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # Vercel request bodies are limited to about 4.5 MB


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


SYSTEM = """You are the voice assistant of ChemSafe AI, used by people with little chemical
training: farmers, small shop owners, students and small-lab workers.
You may receive a photo of a chemical product label and/or a spoken question.

Do this:
1. Read the product name and the active ingredients printed on the label. Only use names that are
   actually visible. Never guess a chemical that you cannot read.
2. Answer in {language}. Use short, simple, spoken-style sentences (about 120 words at most).
   No markdown, no tables, no emojis, no lists with symbols.
3. Cover what matters for safety: main hazards (from the label or well-established facts about the
   named chemical), safe storage, protective equipment, what NOT to store or mix it with, basic first
   aid (rinse eyes or skin with plenty of water, move to fresh air, do not induce vomiting unless the
   label says so), and when to call a doctor or the local poison helpline.
4. Answer the person's question directly if they asked one.
5. NEVER give spraying doses, dilution ratios, application rates or mixing recipes. Say to follow the
   printed label and ask the local agriculture or safety officer.
6. If the photo is unreadable or no chemical can be identified, say so kindly in {language}, ask for a
   clearer photo, and set readable to false. Never invent information. If unsure, say you are unsure.

Reply with ONLY a JSON object, no other text:
{{"chemicals": ["English chemical or active-ingredient names for database lookup, at most 3"],
  "readable": true or false,
  "answer": "the answer written in {language}"}}"""


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        self._send(200, {"ok": True, "model": MODEL, "key_set": bool(api_key())})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "Bad request."})

        question = (d.get("question") or "").strip()[:1000]
        image = d.get("image")
        language = str(d.get("language") or "English")[:40]
        if not image and not question:
            return self._send(400, {"error": "Send a photo or a question."})
        if not api_key():
            return self._send(500, {"error": "Server is missing GEMINI_API_KEY. Add it in Vercel settings and redeploy."})

        img_bytes = None
        if image:
            try:
                img_bytes = base64.b64decode(image, validate=True)
            except Exception:
                return self._send(400, {"error": "The photo could not be read."})
            if len(img_bytes) > MAX_IMAGE_BYTES:
                return self._send(413, {"error": "Photo is too large. Try a smaller photo."})

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key())
            parts = []
            if img_bytes:
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            parts.append(question or "Please read this label and tell me how to handle it safely.")
            resp = client.models.generate_content(
                model=MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM.format(language=language),
                    response_mime_type="application/json",
                    max_output_tokens=2048,
                ),
            )
            raw = (resp.text or "").strip()
            if not raw:
                return self._send(502, {"error": "The AI gave no answer. Try a clearer photo or rephrase the question."})
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            data = json.loads(raw)
            chems = [str(c)[:80] for c in (data.get("chemicals") or [])][:3]
            return self._send(200, {"chemicals": chems,
                                    "readable": bool(data.get("readable", True)),
                                    "answer": str(data.get("answer", ""))[:2000]})
        except json.JSONDecodeError:
            return self._send(502, {"error": "The AI reply could not be understood. Please try again."})
        except Exception as e:
            return self._send(502, {"error": "AI service error: " + str(getattr(e, "message", e))[:200]})
