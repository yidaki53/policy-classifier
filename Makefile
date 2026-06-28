ROOT?=$(shell pwd)

.PHONY: incremental-update figures manuscript

incremental-update:
	cd "$(ROOT)" && uv run python scripts/update_pipeline.py --cpu-fraction 0.25
	$(MAKE) -C "$(ROOT)/manuscript" figures
	$(MAKE) -C "$(ROOT)/manuscript" manuscript

figures:
	$(MAKE) -C "$(ROOT)/manuscript" figures

manuscript:
	$(MAKE) -C "$(ROOT)/manuscript" manuscript