# AGENTS.md — Car-Type-Classification-Service

**Repo:** https://github.com/AlphaeusNg/Car-Type-Classification-Service  
**Local:** `/home/alph/projects/Car-Type-Classification-Service`  
**Hub:** `/home/alph/projects/AGENTS.md`  
**Kind:** ML service (not GitHub Pages)

## Purpose

Classify car images into **196 Stanford Cars** types via ResNet50 transfer learning. Exposes a FastAPI `/predict` endpoint; includes training notebooks and Docker packaging.

## Structure

```text
api/
  main.py              # FastAPI app
  utils.py             # Load model, preprocess, predict
run.py                 # Setup / launch helper
requirements.txt
Dockerfile
model_training.ipynb
gpu_setup_guide.ipynb
prediction_example.py
class_mapping.json
best_car_model.keras / car_classification_model.h5 / models/
data/train  data/test  # Stanford Cars images (large)
```

## Runtime notes

- Python **3.12+** recommended; TensorFlow ~2.19 in project history.
- Models and `data/` are heavy — don’t casually reformat or re-upload huge assets.
- Prefer API changes in `api/` with small utilities; keep training experiments in notebooks unless promoting a new saved model.

## Commands

```bash
cd /home/alph/projects/Car-Type-Classification-Service

# Virtualenv (if used locally)
# python3 -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

python3 run.py            # project’s one-command helper (see README)
# or uvicorn via API module as documented in README

python3 prediction_example.py
docker build -t car-type-clf .
```

## Conventions

- `snake_case` Python; 4-space indent.
- Don’t commit secrets or personal dataset paths.
- If wiring this into the portfolio as a “project card”, edit **alphaeusng.github.io** modals/copy separately — this repo stays the service.

## Relation to other projects

Standalone research/engineering artifact. No runtime dependency on AlpArcade, VerseKeep, or the portfolio static sites.

## Agent checklist

1. Confirm GPU/CPU expectations before retraining.
2. Keep API contract stable or document breaking changes.
3. Avoid deleting `data/` or model weights without explicit user request.
4. Push only this remote; don’t mix with static-site commits.
