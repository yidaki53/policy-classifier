from swedish_parliament_policy_classifier.io.markdown_frontmatter import ensure_frontmatter, has_frontmatter


def test_ensure_frontmatter_prepends_when_missing():
    text = "# Title\n\nBody\n"
    out = ensure_frontmatter(text, {"_agent_frontmatter": {"id": "doc.test"}})
    assert has_frontmatter(out)
    assert "_agent_frontmatter:" in out
    assert "id: doc.test" in out
    assert "# Title" in out


def test_ensure_frontmatter_preserves_existing_frontmatter():
    text = "---\nsection_id: test\n---\n\n# Title\n"
    out = ensure_frontmatter(text, {"_agent_frontmatter": {"id": "doc.test"}})
    assert out == text
