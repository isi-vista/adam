default:
	@echo "an explicit target is required"

# easier to test python2 vs. python3
PYTHON=python3

SHELL=bash
SOURCE_DIR_NAME=adam

PYLINT:=pylint $(SOURCE_DIR_NAME) tests benchmarks

MYPY:=mypy $(MYPY_ARGS) $(SOURCE_DIR_NAME) tests benchmarks

# Suppressed warnings:
# Too many arguments, Unexpected keyword arguments: can't do static analysis on attrs __init__
# Signature of "__getitem__": https://github.com/python/mypy/issues/4108
# Module has no attribute *: mypy doesn't understand __init__.py imports
# mypy/typeshed/stdlib/3/builtins.pyi:39: This is evidence given for false positives on
#   attrs __init__ methods. (This line is for object.__init__.)
# X has no attribute "validator" - thrown for mypy validator decorators, which are dynamically generated
# X has no attribute "default" - thrown for mypy default decorators, which are dynamically generated
# SelfType" has no attribute - mypy seems not to be able to figure out the methods of self for SelfType
#
# we tee the output to a file and test if it is empty so that we can return an error exit code
# if there are errors (for CI) while still displaying them to the user
FILTERED_MYPY:=$(MYPY) | perl -ne 'print if !/(Too many arguments|Signature of "__getitem__"|Only concrete class|Unexpected keyword argument|mypy\/typeshed\/stdlib\/3\/builtins.pyi:39: note: "\w+" defined here|Module( '\''\w+'\'')? has no attribute|has no attribute "validator"|has no attribute "default"|SelfType" has no attribute)/' | tee ./.mypy_tmp && test ! -s ./.mypy_tmp  

# this is the standard ignore list plus ignores for hanging indents, pending figuring out how to auto-format them
FLAKE8:=flake8
FLAKE8_CMD:=$(FLAKE8) $(SOURCE_DIR_NAME)

test: 
	pytest tests

coverage:
	pytest --cov tests

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

check: black-check lint mypy flake8

benchmark:
	pytest benchmarks --benchmark-timer=time.process_time

precommit: black-fix check
