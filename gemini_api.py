import base64
import json
import requests


def _list_generate_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"ListModels failed: {response.status_code} {response.text}")
    data = response.json()
    models = []
    for model in data.get("models", []):
        methods = model.get("supportedGenerationMethods", [])
        name = model.get("name", "")
        if "generateContent" in methods and name.startswith("models/"):
            models.append(name.replace("models/", "", 1))
    return models


def _extract_text(result_json):
    try:
        parts = result_json["candidates"][0]["content"]["parts"]
        return "\n".join([p.get("text", "") for p in parts if "text" in p]).strip()
    except Exception:
        return ""


def _gemini_generate(api_key, parts):
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")

    available_models = _list_generate_models(api_key)
    if not available_models:
        raise RuntimeError("No Gemini model with generateContent support was returned for this API key.")

    preferred_order = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-pro-vision",
    ]
    model_candidates = [m for m in preferred_order if m in available_models]
    for model_name in available_models:
        if model_name not in model_candidates:
            model_candidates.append(model_name)

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": parts}]}
    last_error = None

    for model_name in model_candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        if response.status_code == 200:
            result_json = response.json()
            text = _extract_text(result_json)
            if text:
                return text
            last_error = "Gemini response did not contain text output."
            continue
        last_error = f"{response.status_code} {response.text}"

    raise RuntimeError(f"Gemini request failed for all configured models. Last error: {last_error}")


def gemini_analyze_leaf_health(image_bytes, api_key):
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "You are a plant pathology assistant. Analyze the given plant leaf image and return ONLY valid JSON.\n"
        "Output schema:\n"
        "{\n"
        '  "plant_name": "string",\n'
        '  "disease_name": "string",\n'
        '  "is_healthy": true,\n'
        '  "confidence_note": "short string",\n'
        '  "detailed_advice": "clear, practical recommendation in 4-6 sentences"\n'
        "}\n"
        "Rules:\n"
        "- is_healthy must be true or false.\n"
        "- If unclear, still provide your best estimate and mention uncertainty in confidence_note.\n"
        "- Do not include markdown or code fences."
    )
    text = _gemini_generate(
        api_key,
        [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
        ],
    )

    try:
        parsed = json.loads(text)
        return {
            "plant_name": parsed.get("plant_name", "Unknown"),
            "disease_name": parsed.get("disease_name", "Unknown"),
            "is_healthy": bool(parsed.get("is_healthy", False)),
            "confidence_note": parsed.get("confidence_note", "Estimated by Gemini."),
            "detailed_advice": parsed.get("detailed_advice", ""),
        }
    except Exception:
        return {
            "plant_name": "Unknown",
            "disease_name": "Unknown",
            "is_healthy": False,
            "confidence_note": "Gemini returned unstructured output.",
            "detailed_advice": text,
        }


def gemini_chat_response(user_message, prediction_context, project_context, api_key):
    context = prediction_context or {}
    site_context = project_context or {}
    prompt = (
        "You are LeafAI assistant. Be friendly, natural, and conversational.\n"
        "You can answer both plant-health questions and website/project questions.\n"
        "If user asks about who made this, how it works, technologies, or purpose, use the provided website context.\n"
        "If user asks non-plant casual questions, answer briefly and then offer help with the app.\n"
        "Avoid robotic numbered templates unless user asks for step-by-step format.\n"
        f"Prediction context (JSON): {json.dumps(context)}\n"
        f"Website context (JSON): {json.dumps(site_context)}\n"
        f"User question: {user_message}\n\n"
        "Response style:\n"
        "- Keep response concise, human, and useful.\n"
        "- Use short paragraphs or bullets only when helpful.\n"
        "- If a prediction context exists, connect answer with it.\n"
        "- If no prediction is available, ask user to upload an image for diagnosis."
    )
    return _gemini_generate(api_key, [{"text": prompt}])
