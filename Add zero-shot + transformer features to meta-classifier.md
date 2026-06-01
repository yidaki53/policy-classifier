# Problem
The LightGBM meta-classifier currently only uses keyword counts + embedding cosine similarities as features (plus topic/metadata). This yields 10.7% test accuracy because these signals are too weak alone.
# Available signal sources not yet in the feature vector
1. **Zero-shot NLI scores** (`zs_map`) -- already computed in `score_motion` Stage 3 but NOT passed to `build_feature_vector`
2. **Transformer ideology classifier** (`models/transformer_ideology_classifier/final`) -- a fine-tuned BertForSequenceClassification with 69.7% accuracy on our exact 7 categories, not used at all
3. The finetuned_swedish_bert is already used as the embedding model (EmbeddingMatcher prefers it), so path 3 is really about the classification head
# Changes
## 1. `ensemble.py` -- extend feature vector
* `_build_feature_names`: add `zs_{cat}` (7 features) and `bert_cls_{cat}` (7 features) blocks
* `build_feature_vector`: add `zero_shot_scores` and `bert_cls_scores` optional dict params
* `prepare_training_data_from_gold_labels`: compute zero-shot scores and transformer predictions per motion
## 2. New helper: `classifier/transformer_predict.py`
* Load `BertForSequenceClassification` from `models/transformer_ideology_classifier/final`
* Expose `predict_proba(text) -> Dict[str, float]` that returns per-category probabilities
* Lazy-load model, cache on first use
## 3. `scorer.py` -- pass new signals to meta-classifier
* Pass existing `zs_map` to `build_feature_vector` (one-line addition)
* Call transformer predictor and pass `bert_cls_scores` to `build_feature_vector`
## 4. `predict_with_meta_classifier` -- already handles missing features
* Pads missing columns with zeros, so old pickles still work
* New pickles will have the extra columns in `_feature_names`
# Expected outcome
The transformer classifier alone achieves 69.7%. Adding its predictions as features plus zero-shot scores should let the meta-classifier reach or exceed that, since it can additionally leverage keyword, embedding, and topic signals.