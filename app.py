# app.py
import re
import traceback
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
import spacy
import joblib
from huggingface_hub import hf_hub_download
import spacy
from spacy.cli import download

app = Flask(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

MODEL_REPO = "divishaupadhyay2/cordibot-task-detector"
MODEL_FILE = "model.joblib"

try:
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    model = joblib.load(model_path)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)

def decode_deadline(text, reference_time=None):
    if not text:
        return None

    if reference_time is None:
        reference_time = datetime.now()

    text = text.lower().strip()

    if text in ["eod", "end of day"]:
        return reference_time.replace(hour=17, minute=0, second=0)

    if text == "cob":
        return reference_time.replace(hour=18, minute=0, second=0)

    if "within the hour" in text:
        return reference_time + timedelta(hours=1)

    time_match = re.search(r"(\d{1,2})\s*(am|pm)", text)
    if time_match:
        hour = int(time_match.group(1))
        meridian = time_match.group(2)
        if meridian == "pm" and hour != 12:
            hour += 12
        if meridian == "am" and hour == 12:
            hour = 0
        return reference_time.replace(hour=hour, minute=0, second=0)

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    for day, idx in weekdays.items():
        if day in text:
            return next_weekday(reference_time, idx).replace(hour=17, minute=0, second=0)

    return None

def extract_linguistic_features(text):
    doc = nlp(text)
    return {
        "verbs": [t.lemma_ for t in doc if t.pos_ == "VERB"],
        "persons": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "dates": [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    }

def extract_assignee(text, feats):
    m = re.search(r"@(\w+)", text)
    if m:
        return f"@{m.group(1)}"

    if re.search(r"\b(you|your)\b", text.lower()):
        return "Addressee"
    if re.search(r"\b(i|me|my)\b", text.lower()):
        return "Speaker"

    if feats["persons"]:
        return feats["persons"][0]

    return "Unassigned"

def extract_deadline(text, feats):
    keywords = ["eod", "today", "tomorrow", "next week", "asap"]
    text_l = text.lower()

    for k in keywords:
        if k in text_l:
            d = decode_deadline(k)
            return k, d.isoformat() if d else None

    if feats["dates"]:
        d = decode_deadline(feats["dates"][0])
        return feats["dates"][0], d.isoformat() if d else None

    return None, None

def clean_description(text, assignee, deadline_text):
    desc = text

    if assignee and assignee != "Unassigned":
        desc = desc.replace(assignee.replace("@", ""), "")
        desc = desc.replace(assignee, "")

    if deadline_text:
        desc = desc.replace(deadline_text, "")

    desc = re.sub(
        r"\b(please|kindly|could|would|can|maybe|perhaps|need to|should|must)\b",
        "",
        desc,
        flags=re.I
    )

    return re.sub(r"\s+", " ", desc).strip(" ,.:;-")


@app.route("/process", methods=["POST"])
def process():
    if nlp is None:
     nlp = spacy.load("en_core_web_sm")

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    

    data = request.get_json() or {}
    text = data.get("message", "").strip()

    if not text:
        return jsonify({"error": "No message provided"}), 400

    try:
        confidence = float(model.predict_proba([text])[0][1])
        is_task = confidence >= 0.35

        if not is_task:
            return jsonify({
                "is_task": bool(is_task),
                "confidence": confidence
            })

        feats = extract_linguistic_features(text)
        assignee = extract_assignee(text, feats)
        deadline_text, deadline_iso = extract_deadline(text, feats)
        description = clean_description(text, assignee, deadline_text)

        return jsonify({
            "is_task": True,
            "description": description,
            "assigned_to": assignee,
            "deadline": deadline_iso if deadline_iso else deadline_text,
            "deadline_raw": deadline_text,
            "confidence": confidence,
            "model": "TF-IDF + Logistic Regression"
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Processing failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "TF-IDF + Logistic Regression",
        "nlp": "spaCy"
    })


if __name__ == "__main__":
    app.run(debug=True)
