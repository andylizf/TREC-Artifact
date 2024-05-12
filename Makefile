PYTHON = python

install:
	$(PYTHON) setup.py install

clean:
	rm -rf build dist *.egg-info __pycache__ 

.PHONY: install clean