default:

githubciinstall:
	pip install netam@git+https://github.com/matsengrp/netam.git
	pip install -e .

install:
	pip install -e .

formatdocs:
	docformatter --in-place --black --recursive dnsmex

format:
	docformatter --in-place --black --recursive dnsmex tests || echo "Docformatter made changes"
	black dnsmex tests

format-commit-push:
	@make format || true
	@if [ -n "$$(git status --porcelain)" ]; then \
		git add -u && \
		git commit -m "make format" && \
		git push; \
	else \
		echo "No formatting changes to commit"; \
		git push; \
	fi

checkformat:
	# docformatter --check --black --recursive dnsmex
	black --check dnsmex tests

checktodo:
	(find . -name "*.py" -o -name "*.Snakemake" | grep -v "/\." | xargs grep -l "TODO") && echo "TODOs found" && exit 1 || echo "No TODOs found" && exit 0

test:
	pytest tests
	cd tests/simulation; ./test_simulation_cli.sh; cd ../..

lint:
	flake8 dnsmex --max-complexity=30 --ignore=E731,W503,E402,F541,E501,E203,E266 --statistics --exclude=__pycache__


runnotebooks:
	./run_notebooks.sh

.PHONY: default install test formatdocs format format-commit-push checkformat checktodo lint runnotebooks githubciinstall
