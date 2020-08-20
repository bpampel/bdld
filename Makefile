.PHONY: clean-build install package venv

VIRTUAL_ENV?=.venv
PY=$(VIRTUAL_ENV)/bin/python3
PIP=$(VIRTUAL_ENV)/bin/pip3

install: venv
	$(PY) -m pip install -e .

package: venv
	$(PIP) install -U pip wheel
	python3 setup.py sdist bdist_wheel

venv: $(VIRTUAL_ENV)/bin/activate
$(VIRTUAL_ENV)/bin/activate: requirements.txt
	test -d .venv || python3 -m venv $(VIRTUAL_ENV)
	$(PIP) install -U pip; $(PIP) install -Ur requirements.txt
	@touch $(VIRTUAL_ENV)/bin/activate

clean-pyc:
	find . -regex '^.*\(__pycache__\|\.py[co]\)' -delete

clean-build:
	rm -rf build*
	rm -rf dist*
	rm -rf *.egg-info

clean-venv:
	rm -r .venv

clean-all: clean-pyc clean-build clean-venv

black-exists: ; @which black > /dev/null
format: black-exists
	black .
