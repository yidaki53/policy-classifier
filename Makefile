ROOT?=$(shell pwd)

.PHONY: incremental-update figures manuscript publication-bundle publication-results external-handoff

incremental-update:
	cd "$(ROOT)" && uv run python scripts/update_pipeline.py --cpu-fraction 0.25
	$(MAKE) -C "$(ROOT)/manuscript" figures
	$(MAKE) -C "$(ROOT)/manuscript" manuscript

external-handoff:
	cd "$(ROOT)" && uv run python scripts/update_pipeline.py --cpu-fraction 0.25
	$(MAKE) -C "$(ROOT)/manuscript" figures
	$(MAKE) -C "$(ROOT)/manuscript" manuscript
	$(MAKE) -C "$(ROOT)/manuscript" anonymized
	$(MAKE) publication-bundle
	cd "$(ROOT)" && uv run python -c "from pathlib import Path; from swedish_parliament_policy_classifier.analysis.publication_workflow import build_external_handoff_package; root = Path.cwd(); build_external_handoff_package(root=root, output_dir=root / 'output' / 'external_handoff', regular_manuscript_path=root / 'manuscript.md', anonymized_manuscript_path=root / 'manuscript_anonymized.md')"

figures:
	$(MAKE) -C "$(ROOT)/manuscript" figures

manuscript:
	$(MAKE) -C "$(ROOT)/manuscript" manuscript

publication-bundle:
	cd "$(ROOT)" && uv run python scripts/build_publication_bundle.py --root "$(ROOT)" --output-dir "$(ROOT)/output/publication_bundle" --tag "submission-local" --title "Publication bundle" --artifact-root manuscript/build --artifact-root figures --artifact-root output/analysis --artifact-root output/manuscript

publication-results:
	test -n "$(EVALUATION)"
	cd "$(ROOT)" && uv run python scripts/build_publication_result_bundle.py --root "$(ROOT)" --evaluation "$(EVALUATION)"