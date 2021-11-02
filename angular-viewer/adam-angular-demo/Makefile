default:
	@echo "an explicit target is required"

SHELL=/usr/bin/env bash

PRETTIER_FILES=. ../../README.md ../../RELEASE_PROCESS.md ../../adam/experiment/README.md

stylelint:
	npx stylelint "src/**/*.css"

lint:
	npx ng lint

prettier-fix:
	npx prettier --write $(PRETTIER_FILES)

prettier-check:
	npx prettier --check $(PRETTIER_FILES)

# Re-enable stylelint once we have a configuration for styles.css
check: prettier-check lint # stylelint

precommit: prettier-fix check

update:
	npm install

install:
	npm install
