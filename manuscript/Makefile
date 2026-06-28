ROOT=..
OUTPUT_DIR=build
SCIREPS_OUTPUT_DIR=build_scientific_reports
SCIREPS_RENDERED_DIR=$(SCIREPS_OUTPUT_DIR)/rendered_sections
SCIREPS_RENDERED_SECTIONS=$(SCIREPS_RENDERED_DIR)/01_title.md $(SCIREPS_RENDERED_DIR)/01a_abstract.md $(SCIREPS_RENDERED_DIR)/02_question.md $(SCIREPS_RENDERED_DIR)/03_methodology.md $(SCIREPS_RENDERED_DIR)/03_results.md $(SCIREPS_RENDERED_DIR)/04_significance.md $(SCIREPS_RENDERED_DIR)/07_appendix.md $(SCIREPS_RENDERED_DIR)/05_data_availability.md $(SCIREPS_RENDERED_DIR)/06_acknowledgments.md
SCIREPS_COMBINED=$(SCIREPS_OUTPUT_DIR)/combined.md
SCIREPS_MANUSCRIPT_MD=$(ROOT)/manuscript_scientific_reports.md
SCIREPS_MANUSCRIPT_PDF=$(ROOT)/manuscript_scientific_reports.pdf
SECTIONS=sections/01_title.md sections/01a_abstract.md sections/02_question.md sections/03_methodology.md sections/03_results.md sections/04_significance.md sections/07_appendix.md sections/05_data_availability.md sections/06_acknowledgments.md
RENDERED_DIR=$(OUTPUT_DIR)/rendered_sections
RENDERED_SECTIONS=$(RENDERED_DIR)/01_title.md $(RENDERED_DIR)/01a_abstract.md $(RENDERED_DIR)/02_question.md $(RENDERED_DIR)/03_methodology.md $(RENDERED_DIR)/03_results.md $(RENDERED_DIR)/04_significance.md $(RENDERED_DIR)/07_appendix.md $(RENDERED_DIR)/05_data_availability.md $(RENDERED_DIR)/06_acknowledgments.md
COMBINED=$(OUTPUT_DIR)/combined.md
ANON_COMBINED=$(OUTPUT_DIR)/combined_anonymized.md
MANUSCRIPT_MD=$(ROOT)/manuscript.md
ANON_MANUSCRIPT_MD=$(ROOT)/manuscript_anonymized.md
MANUSCRIPT_PDF=$(ROOT)/manuscript.pdf
ANON_MANUSCRIPT_PDF=$(ROOT)/manuscript_anonymized.pdf
BIB=$(ROOT)/manuscript/bibliography/references.bib
CSL=$(ROOT)/manuscript/bibliography/nature.csl
SCIREPS_CSL=$(ROOT)/manuscript/bibliography/nature.csl

PANDOC_OPTS=--standalone --citeproc --bibliography=$(BIB) --csl=$(CSL)

.PHONY: all figures manuscript pdf docx html combined anonymized anonymized-pdf check-frontmatter render journal-check stats clean incremental-update polish

all: figures manuscript

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

figures:
	cd $(ROOT) && uv run python scripts/visualize.py --profiles data/parquet/party_profiles_recency.parquet --out figures
	cd $(ROOT) && uv run python scripts/visualize_voting.py --votering-parquet data/votering/parquet --out figures/voting

combined: render polish | $(OUTPUT_DIR)
	@echo "Creating combined manuscript file (frontmatter-safe)..."
	@: > $(COMBINED)
	@for f in $(RENDERED_SECTIONS); do \
		awk 'BEGIN{in_fm=0;first=1} \
		{ \
			if (first && $$0 == "---") { in_fm=1; first=0; next } \
			if (in_fm && $$0 == "---") { in_fm=0; next } \
			first=0; \
			if (!in_fm) print $$0; \
		}' $$f >> $(COMBINED); \
		echo "" >> $(COMBINED); \
	done
	@if grep -Eq '\{\{[^}]+\}\}' $(COMBINED); then \
		echo "Unresolved template token(s) detected in $(COMBINED)"; \
		exit 1; \
	fi
	@cp $(COMBINED) $(MANUSCRIPT_MD)

render: check-frontmatter | $(OUTPUT_DIR)
	uv run python $(ROOT)/scripts/render_manuscript_jinja.py --repo-root $(ROOT) --manuscript-dir manuscript --out-dir manuscript/build/rendered_sections --context-out manuscript/build/manuscript_context.json

journal-check: | $(OUTPUT_DIR)
	uv run python $(ROOT)/scripts/check_journal_requirements.py --repo-root $(ROOT) --manuscript-dir manuscript --journal-profile manuscript/journal_profiles/plos_one.yaml --out manuscript/build/journal_requirements_report.json

pdf: render combined
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc $(COMBINED) $(PANDOC_OPTS) -o $(OUTPUT_DIR)/manuscript.pdf; \
		cp $(OUTPUT_DIR)/manuscript.pdf $(MANUSCRIPT_PDF); \
		echo "PDF created: $(OUTPUT_DIR)/manuscript.pdf"; \
	else \
		echo "pandoc not found; skipping PDF build"; \
	fi

docx: render combined
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc $(COMBINED) $(PANDOC_OPTS) -o $(OUTPUT_DIR)/manuscript.docx; \
		echo "DOCX created: $(OUTPUT_DIR)/manuscript.docx"; \
	else \
		echo "pandoc not found; skipping DOCX build"; \
	fi

html: render combined
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc $(COMBINED) $(PANDOC_OPTS) -o $(OUTPUT_DIR)/manuscript.html; \
		echo "HTML created: $(OUTPUT_DIR)/manuscript.html"; \
	else \
		echo "pandoc not found; skipping HTML build"; \
	fi

anonymized: render combined
	uv run python $(ROOT)/scripts/create_anonymized_manuscript.py --in $(COMBINED) --out $(ANON_COMBINED)
	@cp $(ANON_COMBINED) $(ANON_MANUSCRIPT_MD)

anonymized-pdf: anonymized
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc $(ANON_COMBINED) $(PANDOC_OPTS) -o $(OUTPUT_DIR)/manuscript_anonymized.pdf && \
		cp $(OUTPUT_DIR)/manuscript_anonymized.pdf $(ANON_MANUSCRIPT_PDF) && \
		echo "Anonymized PDF created: $(OUTPUT_DIR)/manuscript_anonymized.pdf"; \
	else \
		echo "pandoc not found; skipping anonymized PDF build"; \
	fi

manuscript: pdf

check-frontmatter:
	@echo "Checking section frontmatter..."
	@for f in $(SECTIONS); do \
		if ! head -n 1 $$f | grep -q '^---$$'; then \
			echo "Missing YAML frontmatter start in $$f"; exit 1; \
		fi; \
		if ! grep -q '^section_id:' $$f; then \
			echo "Missing section_id in $$f"; exit 1; \
		fi; \
		if ! grep -q '^section_title:' $$f; then \
			echo "Missing section_title in $$f"; exit 1; \
		fi; \
	done
	@echo "Frontmatter check passed."

stats: combined
	@echo "Word count (combined markdown):"
	@wc -w $(COMBINED)

incremental-update: polish
	cd $(ROOT) && uv run python scripts/update_pipeline.py --cpu-fraction 0.25

scireps-render: check-frontmatter | $(SCIREPS_OUTPUT_DIR)
	uv run python $(ROOT)/scripts/render_manuscript_jinja.py --repo-root $(ROOT) --manuscript-dir manuscript --out-dir $(SCIREPS_RENDERED_DIR) --context-out $(SCIREPS_OUTPUT_DIR)/manuscript_context.json --journal-profile manuscript/journal_profiles/scientific_reports.yaml

scireps-combined: scireps-render | $(SCIREPS_OUTPUT_DIR)
	@echo "Creating Scientific Reports combined manuscript file..."
	@: > $(SCIREPS_COMBINED)
	@for f in $(SCIREPS_RENDERED_SECTIONS); do \
		awk 'BEGIN{in_fm=0;first=1} \
		{ \
			if (first && $$0 == "---") { in_fm=1; first=0; next } \
			if (in_fm && $$0 == "---") { in_fm=0; next } \
			first=0; \
			if (!in_fm) print $$0; \
		}' $$f >> $(SCIREPS_COMBINED); \
		echo "" >> $(SCIREPS_COMBINED); \
	done
	@if grep -Eq '\{\{[^}]+\}\}' $(SCIREPS_COMBINED); then \
		echo "Unresolved template token(s) detected in $(SCIREPS_COMBINED)"; \
		exit 1; \
	fi
	@cp $(SCIREPS_COMBINED) $(SCIREPS_MANUSCRIPT_MD)

scireps-pdf: scireps-combined
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc $(SCIREPS_COMBINED) $(PANDOC_OPTS) -o $(SCIREPS_OUTPUT_DIR)/manuscript.pdf; \
		cp $(SCIREPS_OUTPUT_DIR)/manuscript.pdf $(SCIREPS_MANUSCRIPT_PDF); \
		echo "Scientific Reports PDF created: $(SCIREPS_OUTPUT_DIR)/manuscript.pdf"; \
	else \
		echo "pandoc not found; skipping Scientific Reports PDF build"; \
	fi

scireps-journal-check: | $(SCIREPS_OUTPUT_DIR)
	uv run python $(ROOT)/scripts/check_journal_requirements.py --repo-root $(ROOT) --manuscript-dir manuscript --journal-profile manuscript/journal_profiles/scientific_reports.yaml --out $(SCIREPS_OUTPUT_DIR)/journal_requirements_report.json

scireps-clean:
	rm -rf $(SCIREPS_OUTPUT_DIR) $(SCIREPS_MANUSCRIPT_MD) $(SCIREPS_MANUSCRIPT_PDF)

clean: scireps-clean
	rm -rf $(OUTPUT_DIR) $(MANUSCRIPT_MD) $(MANUSCRIPT_PDF) $(ANON_MANUSCRIPT_MD) $(ANON_MANUSCRIPT_PDF)

polish:
	cd $(ROOT) && uv run python scripts/polish_manuscript_text.py --dir sections
