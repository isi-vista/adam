default:
	@echo "an explicit target is required"

# easier to test python2 vs. python3
PYTHON=pypy3

SHELL=bash
SOURCE_DIR_NAME=adam

PYLINT:=pylint $(SOURCE_DIR_NAME) tests benchmarks

MYPY:=mypy $(MYPY_ARGS) $(SOURCE_DIR_NAME) tests benchmarks

# Suppressed warnings:
# Too many arguments, Unexpected keyword arguments: can't do static analysis on attrs __init__
# Module has no attribute *: mypy doesn't understand __init__.py imports
# mypy/typeshed/stdlib/3/builtins.pyi:39: This is evidence given for false positives on
#   attrs __init__ methods. (This line is for object.__init__.)
# X has no attribute "validator" - thrown for mypy validator decorators, which are dynamically generated
# X has no attribute "default" - thrown for mypy default decorators, which are dynamically generated
# SelfType" has no attribute - mypy seems not to be able to figure out the methods of self for SelfType
#
# we tee the output to a file and test if it is empty so that we can return an error exit code
# if there are errors (for CI) while still displaying them to the user
FILTERED_MYPY:=$(MYPY) | perl -ne 'print if !/(Too many arguments|Only concrete class|Unexpected keyword argument|mypy\/typeshed\/stdlib\/3\/builtins.pyi:39: note: "\w+" defined here|Module( '\''\w+'\'')? has no attribute|has no attribute "validator"|has no attribute "default"|SelfType" has no attribute|Found)/' | tee ./.mypy_tmp && test ! -s ./.mypy_tmp

# this is the standard ignore list plus ignores for hanging indents, pending figuring out how to auto-format them
FLAKE8:=flake8
FLAKE8_CMD:=$(FLAKE8) $(SOURCE_DIR_NAME)

IGNORE_TESTS = --ignore tests/experiment_test.py --ignore tests/continuous_test.py

test: 
	$(PYTHON) -m pytest $(IGNORE_TESTS) tests

coverage:
	$(PYTHON) -m pytest $(IGNORE_TESTS) --cov=adam tests

benchmark:
	$(PYTHON) -m pytest $(IGNORE_TESTS) benchmarks

lint:
	$(PYLINT)

mypy:
	$(FILTERED_MYPY)

flake8:
	$(FLAKE8_CMD)

black-fix:
	black $(SOURCE_DIR_NAME) tests benchmarks

black-check:
	black --check $(SOURCE_DIR_NAME) tests benchmarks

doc-lint:
	sphinx-build -nT -b dummy docs docs/_build/html
# treating warnings as errors temporarily disabled due to problems
# with circular imports with Relation and Region + Axes
#	sphinx-build -nWT -b dummy docs docs/_build/html

reqs-freeze:
	pip freeze >requirements_lock.txt

check: black-check mypy flake8 doc-lint lint

precommit: reqs-freeze black-fix check
