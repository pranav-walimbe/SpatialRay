.DEFAULT_GOAL := fix

REGION ?= us-west-2

.PHONY: sync fix lint test ci perf clear-ec2

sync:
	uv sync --group dev

fix:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest || [ $$? -eq 5 ]

ci: lint test

# default cloud run
perf:
	uv run python -m perf.cloud.launch \
	  --hardware gpu \
	  --model prithvi_eo_v1_100m \
	  --requests 1000 \
	  --rate 5

# terminate all spatialray-perf instances
clear-ec2:
	@regions=$$(aws ec2 describe-regions --all-regions \
	  --query "Regions[?OptInStatus!='not-opted-in'].RegionName" --output text); \
	found=0; \
	for r in $$regions; do \
	  ids=$$(aws ec2 describe-instances --region $$r \
	    --filters "Name=tag:Project,Values=spatialray-perf" \
	      "Name=instance-state-name,Values=pending,running,stopping,stopped,rebooting" \
	    --query "Reservations[].Instances[].InstanceId" --output text); \
	  if [ -n "$$ids" ]; then \
	    found=1; echo "$$r terminating:$$ids"; \
	    aws ec2 terminate-instances --region $$r --instance-ids $$ids >/dev/null; fi; \
	done; \
	if [ $$found -eq 0 ]; then echo "no spatialray-perf instances in any region"; fi
