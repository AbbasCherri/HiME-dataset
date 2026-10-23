DC ?= docker compose

.PHONY: image lint check test-unit test build verify report lock shell

image:
	$(DC) build

lint:
	$(DC) run --rm lint

test-unit:
	$(DC) run --rm test-unit

check: lint test-unit

test:
	$(DC) run --rm test

build:
	$(DC) run --rm pipeline python -m notodata build

verify:
	$(DC) run --rm verify

report:
	$(DC) run --rm pipeline python -m notodata report

lock:
	$(DC) run --rm shell uv lock
	$(DC) run --rm shell uv export --frozen --hashes --no-dev --no-emit-project -o requirements.lock

shell:
	$(DC) run --rm shell
