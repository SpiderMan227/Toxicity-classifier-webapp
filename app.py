from flask import Flask, render_template, request, jsonify, session
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import re
import os
from datetime import datetime

# Minimal preprocess_text copied from notebook
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text.strip()

MODEL_DIR = "toxic_model"
TOKENIZER_DIR = "toxic_tokenizer"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 128
THRESHOLD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR if os.path.isdir(TOKENIZER_DIR) else MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

app = Flask(__name__)
app.secret_key = "your-secret-key-change-this"  # for session

def get_toxicity_class(max_prob):
    """Classify based on max probability across all labels"""
    # new boundaries: Safe [0, 0.35), Moderately Toxic [0.35, 0.65), Toxic [0.65, 1]
    if max_prob < 0.35:
        return "Safe"
    elif max_prob < 0.65:
        return "Moderately Toxic"
    else:
        return "Toxic"

@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    threshold = THRESHOLD
    labels = LABEL_COLS
    probs = None
    preds = None
    toxicity_class = None
    max_prob = None
    
    if request.method == "POST":
        input_text = request.form.get("comment", "").strip()
        # Read slider value as percentage (10..90) and convert to 0-1 fraction.
        threshold_str = request.form.get("threshold", None)
        if threshold_str is not None and threshold_str != "":
            try:
                threshold = float(threshold_str) / 100.0
            except ValueError:
                threshold = THRESHOLD
        else:
            threshold = THRESHOLD
        
        pre = preprocess_text(input_text)
        enc = tokenizer(pre, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()[0]
            probs_arr = 1 / (1 + np.exp(-logits))
            preds_arr = (probs_arr >= threshold).astype(int).tolist()
        
        probs = probs_arr.tolist()
        preds = preds_arr
        max_prob = float(np.max(probs_arr))
        toxicity_class = get_toxicity_class(max_prob)
        
        # Track history in session
        if "history" not in session:
            session["history"] = []
        session["history"].append({
            "text": input_text,
            "toxicity_class": toxicity_class,
            "max_prob": max_prob,
            "timestamp": datetime.now().isoformat()
        })
        session.modified = True
    
    return render_template("index.html", 
                          input_text=input_text, 
                          threshold=threshold,
                          labels=labels,
                          probs=probs,
                          preds=preds,
                          toxicity_class=toxicity_class,
                          max_prob=max_prob)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    text = data.get("text", "")
    pre = preprocess_text(text)
    enc = tokenizer(pre, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= THRESHOLD).astype(int)
    return jsonify({
        "text": text,
        "probabilities": {l: float(p) for l,p in zip(LABEL_COLS, probs)},
        "preds": {l: int(p) for l,p in zip(LABEL_COLS, preds)},
        "threshold": THRESHOLD,
    })

@app.route("/history")
def get_history():
    """Return history data for pie chart and table"""
    hist = session.get("history", [])
    
    # Count by toxicity class
    counts = {"Safe": 0, "Moderately Toxic": 0, "Toxic": 0}
    for entry in hist:
        counts[entry["toxicity_class"]] += 1
    
    return jsonify({
        "counts": counts,
        "history": hist
    })

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear prediction history"""
    session["history"] = []
    session.modified = True
    return jsonify({"success": True})

@app.route("/history_page")
def history_page():
    """Render history page"""
    return render_template("history.html")

if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=5000, debug=False)