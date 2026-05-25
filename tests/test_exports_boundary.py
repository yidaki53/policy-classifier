import importlib

exports = importlib.import_module("swedish_parliament_policy_classifier.exports")


def test_exports_interface():
    # Ensure all canonical exports are present and callable
    assert hasattr(exports, "load_definitions")
    assert callable(exports.load_definitions)
    assert hasattr(exports, "score_motion")
    assert callable(exports.score_motion)
    assert hasattr(exports, "record_lineage")
    assert callable(exports.record_lineage)
    assert hasattr(exports, "persist_classification")
    assert callable(exports.persist_classification)
    assert hasattr(exports, "persist_classifications_batch")
    assert callable(exports.persist_classifications_batch)
    assert hasattr(exports, "save_annotation")
    assert callable(exports.save_annotation)
    assert hasattr(exports, "get_annotation_by_motion")
    assert callable(exports.get_annotation_by_motion)
    assert hasattr(exports, "get_next_unlabeled_motion")
    assert callable(exports.get_next_unlabeled_motion)
    assert hasattr(exports, "get_connection")
    assert callable(exports.get_connection)
    assert hasattr(exports, "init_db")
    assert callable(exports.init_db)
    # Check model types
    assert hasattr(exports, "CategoryDef")
    assert hasattr(exports, "ClassificationResult")
    assert hasattr(exports, "NormalizedMotion")
    assert hasattr(exports, "RawMotion")
    assert hasattr(exports, "PartyProfile")
    # Check NLP helpers
    assert hasattr(exports, "preprocess_text")
    assert callable(exports.preprocess_text)
    assert hasattr(exports, "init_spacy")
    assert callable(exports.init_spacy)
