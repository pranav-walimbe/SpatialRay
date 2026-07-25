.DEFAULT_GOAL := fix

REGION ?= us-west-2

.PHONY: sync fix lint test ci perf-preprocess perf-serve clear-ec2

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

perf-preprocess:
	uv run python -m perf.preprocess

perf-serve:
	uv run --extra perf python -m perf.serve

# terminate running ec2 instances
clear-ec2:
	@ids=$$(aws ec2 describe-instances --region $(REGION) \
	  --filters "Name=tag:Project,Values=spatialray-perf" "Name=instance-state-name,Values=pending,running" \
	  --query "Reservations[].Instances[].InstanceId" --output text); \
	if [ -z "$$ids" ]; then echo "no running spatialray-perf instances in $(REGION)"; else \
	  echo "terminating:$$ids"; \
	  aws ec2 terminate-instances --region $(REGION) --instance-ids $$ids; fi
