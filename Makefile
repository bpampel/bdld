.PHONY: clean-build coverage install package test docs-html docs-pdf

VIRTUAL_ENV?=.venv
PY=$(VIRTUAL_ENV)/bin/python3
PIP=$(VIRTUAL_ENV)/bin/pip3

install: venv
	$(PY) -m pip install -e .

package: venv
	$(PIP) install -U pip wheel
	$(PY) setup.py sdist bdist_wheel

venv: $(VIRTUAL_ENV)/bin/activate
$(VIRTUAL_ENV)/bin/activate: requirements.txt
	test -d $(VIRTUAL_ENV) || python3 -m venv $(VIRTUAL_ENV)
	$(PIP) install -U pip; $(PIP) install -Ur requirements.txt
	@touch $(VIRTUAL_ENV)/bin/activate

docs-html: docs
	@cd docs && make html

docs-pdf: docs
	@cd docs && make latexpdf

test:
	python3 -m unittest discover

coverage: coverage_installed
	@coverage run -m unittest discover > /dev/null
	@coverage report

clean-pyc:
	find . -regex '^.*\(__pycache__\|\.py[co]\)' -delete

clean-build:
	rm -rf build*
	rm -rf dist*
	rm -rf *.egg-info

clean-venv:
	rm -r .venv

clean-all: clean-pyc clean-build clean-venv

format: black_installed
	black .

# check if tool is installed
tools_installed := black_installed coverage_installed
$(tools_installed):
	@which ${@:_installed=} > /dev/null
