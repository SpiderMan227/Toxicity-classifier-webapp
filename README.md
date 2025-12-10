# Toxicity Detector

Lightweight multi-label toxicity detection web app using PyTorch + HuggingFace DistilBERT and Flask.

Features
- Multi-label toxicity predictions (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- Simple web UI (analyze text, view results)
- Session history and summary dashboard
- Dark mode and responsive UI

Quick start (local)
1. Clone:
   git clone <your-repo-url>
   cd "f:\ML_projects\Toxic chat"

2. Create virtualenv and install:
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt

3. Prepare model/tokenizer:
   - Place fine-tuned model into `toxic_model/` and tokenizer into `toxic_tokenizer/`
   - Or update `app.py` to load model path you prefer.

4. Run the app:
   python app.py
   Open http://localhost:5000

API
- POST /api/predict
  Request JSON: { "text": "your text" }
  Response: probabilities and preds per label

Development
- Add tests under tests/ and GitHub Actions will run install step on push.
- Use `session` history stored in Flask session for quick demo (not persistent).

Repository layout
- app.py — Flask app
- templates/ — Jinja2 templates (index.html, history.html)
- toxic_model/, toxic_tokenizer/ — saved model & tokenizer (ignored by git)
- requirements.txt — Python deps
- README.md, LICENSE, CONTRIBUTING.md

License
- MIT (see LICENSE file)

Contact
- Open an issue or create a PR for improvements.
