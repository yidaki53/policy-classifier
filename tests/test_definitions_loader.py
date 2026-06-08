from types import SimpleNamespace

from swedish_parliament_policy_classifier.definitions import loader


def test_recheck_injects_agent_frontmatter_when_missing(tmp_path):
    yaml_path = tmp_path / "political_spectrum.yaml"
    yaml_path.write_text(
        """
schema_version: "2.0.0"
checksum: "PLACEHOLDER"
categories:
  - name: centre
    definition: "x"
    keywords: ["x"]
    regexes: ["x"]
""".lstrip(),
        encoding="utf-8",
    )

    rc = loader._cmd_recheck(SimpleNamespace(file=str(yaml_path)))
    assert rc == 0

    text = yaml_path.read_text(encoding="utf-8")
    assert "_agent_frontmatter:" in text
    assert "id: \"definitions.political_spectrum\"" in text
    assert "checksum: \"PLACEHOLDER\"" not in text
    assert loader.verify(yaml_path)


def test_recheck_preserves_existing_agent_frontmatter(tmp_path):
    yaml_path = tmp_path / "political_spectrum.yaml"
    yaml_path.write_text(
        """
schema_version: "2.0.0"
checksum: "PLACEHOLDER"
_agent_frontmatter:
  id: "definitions.political_spectrum"
  purpose: "kept"
categories:
  - name: centre
    definition: "x"
    keywords: ["x"]
    regexes: ["x"]
""".lstrip(),
        encoding="utf-8",
    )

    rc = loader._cmd_recheck(SimpleNamespace(file=str(yaml_path)))
    assert rc == 0

    text = yaml_path.read_text(encoding="utf-8")
    assert text.count("_agent_frontmatter:") == 1
    assert "purpose: \"kept\"" in text
    assert loader.verify(yaml_path)
