ROOT=..

.PHONY: all figures manuscript clean

all: figures manuscript

figures:
	PYTHONPATH=$(ROOT)/.. python3 -m swedish_parliament_policy_classifier.scripts.ingest --db $(ROOT)/data/swedish_parliament.db
	PYTHONPATH=$(ROOT)/.. python3 -m swedish_parliament_policy_classifier.scripts.classify --db $(ROOT)/data/swedish_parliament.db
	PYTHONPATH=$(ROOT)/.. python3 -m swedish_parliament_policy_classifier.scripts.visualize --db $(ROOT)/data/swedish_parliament.db --out $(ROOT)/figures
	PYTHONPATH=$(ROOT)/.. python3 -m swedish_parliament_policy_classifier.scripts.visualize_advanced --db $(ROOT)/data/swedish_parliament.db --out $(ROOT)/figures
	PYTHONPATH=$(ROOT)/.. python3 -m swedish_parliament_policy_classifier.scripts.visualize_interactive --db $(ROOT)/data/swedish_parliament.db --out $(ROOT)/figures

manuscript:
	cat sections/01_title.md sections/02_question.md sections/03_results.md sections/04_significance.md > ../manuscript.md
	@if command -v pandoc >/dev/null 2>&1; then pandoc ../manuscript.md -o ../manuscript.pdf; fi

clean:
	rm -f ../manuscript.md ../manuscript.pdf
