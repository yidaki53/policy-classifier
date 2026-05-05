# NLP Improvements and roadmap

This folder describes prioritized improvements to the project's NLP pipeline and how to enable optional features.

Priorities

1. Advanced preprocessing: Swedish tokenization, lemmatization and stopword removal (spaCy).  
2. Semantic matching: sentence-transformers embeddings to match motions to category definitions.  
3. Supervised fallback: train a classifier on deterministic labels and calibrate its probabilities.  
4. Human-in-the-loop labeling: active learning CLI to curate edge cases and improve supervised model.  
5. Explainability: integrate LIME/SHAP to explain classifier decisions and surface matched rules.  

Optional dependencies (install in project venv):

```
pip install -r requirements.txt
pip install "spacy" "sentence-transformers"
# install Swedish spaCy model: python -m spacy download sv_core_news_sm
```

Design notes

- All heavy-weight components are optional. The codebase provides safe fallbacks when `spaCy` or `sentence-transformers` are not installed.
- Prefer deterministic rules for transparency; use embeddings and supervised models to resolve ambiguous cases or to suggest labels for human review.
- Persist any new model artifacts and ensure lineage entries are created for model-produced labels.
