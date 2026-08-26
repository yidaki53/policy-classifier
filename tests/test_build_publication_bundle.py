import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_publication_bundle.py"
    spec = importlib.util.spec_from_file_location("build_publication_bundle_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_publication_bundle_creates_manifest_and_archive(tmp_path):
    module = _load_module()

    manuscript_dir = tmp_path / "manuscript" / "build"
    manuscript_dir.mkdir(parents=True)
    (manuscript_dir / "combined.md").write_text("# Manuscript\n", encoding="utf-8")

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "dashboard.html").write_text("<html></html>", encoding="utf-8")

    output_dir = tmp_path / "output" / "release"
    output_dir.mkdir(parents=True)
    (output_dir / "notes.txt").write_text("release note", encoding="utf-8")

    bundle_dir = tmp_path / "bundles"
    manifest = module.build_publication_bundle(
        root=tmp_path,
        output_dir=bundle_dir,
        tag="submission-test",
        commit_sha="abc123",
        title="Publication bundle",
        artifact_roots=["manuscript/build", "figures", "output/release"],
    )

    assert manifest["tag"] == "submission-test"
    assert manifest["commit_sha"] == "abc123"
    assert manifest["title"] == "Publication bundle"
    assert manifest["file_count"] >= 3

    manifest_path = bundle_dir / "publication_bundle_manifest.json"
    archive_path = bundle_dir / "publication_bundle.tar.gz"
    assert manifest_path.exists()
    assert archive_path.exists()

    written_manifest = manifest_path.read_text(encoding="utf-8")
    assert "combined.md" in written_manifest
    assert "dashboard.html" in written_manifest
