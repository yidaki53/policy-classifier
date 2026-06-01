# swedish-parliament-policy-classifier

Prototype project to ingest Swedish Riksdag motions/proposals, classify them on a left–right spectrum,
and produce reproducible analyses and publication-quality figures. This scaffold includes:

- sqlite schema initialization
- Pydantic models
- Riksdag client skeleton (sample mode)
- Deterministic scorer loading YAML definitions
- Ingest script and a simple unit test

Usage (development):

```
python3 -m swedish_parliament_policy_classifier.scripts.ingest --sample
python3 -m pytest -q swedish_parliament_policy_classifier/tests/test_scoring.py
```

Generating visualisations:

```
python3 -m swedish_parliament_policy_classifier.scripts.ingest --db data/swedish_parliament.db
python3 -m swedish_parliament_policy_classifier.scripts.classify --db data/swedish_parliament.db
python3 -m swedish_parliament_policy_classifier.scripts.visualize --db data/swedish_parliament.db --out figures
```

Outputs:
- `data/swedish_parliament.db` — sqlite database with `raw_motions`, `normalized_motions`, `classifications`, and `party_profiles`.
- `figures/party_profiles.png`, `figures/party_profiles.pdf` — stacked-party profile figure.
- `figures/party_profiles_final.png`, `figures/party_profiles_final.pdf` — combined ideological placement + heatmap (publication-ready).

Optional dependencies
---------------------

Some features are optional (heavy ML or language models). Install into the project venv when needed:

```bash
# install base dev deps
pip install -r requirements.txt

# optional NLP and embedding features
pip install spacy sentence-transformers
# install Swedish spaCy model (optional - required for lemmatization)
python -m spacy download sv_core_news_sm
```

The codebase has safe fallbacks if `spaCy` or `sentence-transformers` are not installed; see `nlp/README.md` for details.
