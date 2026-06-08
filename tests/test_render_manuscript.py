import importlib.util
import re
from pathlib import Path

import pandas as pd


def _load_render_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "render_manuscript_jinja.py"
    spec = importlib.util.spec_from_file_location("render_manuscript_jinja_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_sections_have_no_unresolved_template_markers(tmp_path):
    module = _load_render_module()

    repo_root = Path(__file__).resolve().parents[1]
    manuscript_dir = repo_root / "manuscript"
    sections_dir = manuscript_dir / "sections"
    analysis_dir = repo_root / "output" / "analysis"
    journal_profile = manuscript_dir / "journal_profiles" / "plos_one.yaml"
    bibliography = manuscript_dir / "bibliography" / "references.bib"

    context = module._build_context(
        repo_root=repo_root,
        manuscript_dir=manuscript_dir,
        analysis_dir=analysis_dir,
        journal_profile=journal_profile,
        bib_path=bibliography,
    )

    out_dir = tmp_path / "rendered_sections"
    rendered = module._render_sections(sections_dir=sections_dir, out_dir=out_dir, context=context)
    assert rendered, "Expected at least one rendered section"

    unresolved_re = re.compile(r"\{\{[^}]+\}\}|\{%[^%]+%\}")
    unresolved_hits = []
    for item in rendered:
        rendered_path = Path(item["rendered"])
        text = rendered_path.read_text(encoding="utf-8")
        if unresolved_re.search(text):
            unresolved_hits.append(str(rendered_path))

    assert not unresolved_hits, f"Unresolved template markers found in: {unresolved_hits}"


def test_latest_figures_block_uses_image_captions():
    module = _load_render_module()

    repo_root = Path(__file__).resolve().parents[1]
    manuscript_dir = repo_root / "manuscript"
    analysis_dir = repo_root / "output" / "analysis"
    journal_profile = manuscript_dir / "journal_profiles" / "plos_one.yaml"
    bibliography = manuscript_dir / "bibliography" / "references.bib"

    context = module._build_context(
        repo_root=repo_root,
        manuscript_dir=manuscript_dir,
        analysis_dir=analysis_dir,
        journal_profile=journal_profile,
        bib_path=bibliography,
    )

    main_figures_block = context["main_figures_block"]
    appendix_figures_block = context["appendix_figures_block"]

    assert main_figures_block
    assert "![Consistency vs Fulfillment" in main_figures_block
    assert "(updated " in main_figures_block
    assert "Party Modality Overlay" not in main_figures_block

    assert appendix_figures_block
    assert "Party Modality Overlay" in appendix_figures_block
    assert "(updated " in appendix_figures_block
    assert "**Figure 1." not in main_figures_block


def test_promise_fulfillment_example_filters_unknown(tmp_path):
    module = _load_render_module()

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [
            {"party": "SD", "pct_speech_motion_vote": 0.35, "pct_speech_motion_no_vote": 0.01},
            {"party": "L", "pct_speech_motion_vote": 0.19, "pct_speech_motion_no_vote": 0.14},
            {"party": "Unknown", "pct_speech_motion_vote": 0.0, "pct_speech_motion_no_vote": 0.0},
        ]
    ).to_parquet(analysis_dir / "promise_fulfillment_party_summary.parquet", index=False)

    paragraph = module._promise_fulfillment_example_paragraph(analysis_dir)

    assert "`Unknown`" not in paragraph
    assert "`SD`" in paragraph
    assert "`L`" in paragraph


def test_renderer_exclusion_policy_matches_overlay_policy():
    module = _load_render_module()

    assert module.EXCLUDED_COMPARISON_PARTIES == {"Unknown", "Moderaterna", "Vänsterpartiet", "X"}


def test_render_sections_injects_frontmatter_when_source_missing(tmp_path):
    module = _load_render_module()

    sections_dir = tmp_path / "sections"
    sections_dir.mkdir(parents=True)
    (sections_dir / "99_generated.md").write_text("# Example\n\nRendered body.\n", encoding="utf-8")

    out_dir = tmp_path / "rendered"
    rendered = module._render_sections(sections_dir=sections_dir, out_dir=out_dir, context={"generated_utc": "2026-06-08T00:00:00Z"})

    assert len(rendered) == 1
    rendered_path = Path(rendered[0]["rendered"])
    text = rendered_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "_agent_frontmatter:" in text
    assert "source_section:" in text
    assert "# Example" in text
