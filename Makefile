ipynb_files := $(wildcard *.ipynb)

.PHONY: all clean

all: $(ipynb_files)

$(ipynb_files): %.ipynb: %.md
	jupytext --sync $<
	chmod 755 $*.py
	touch $@ $*.py

clean:
	rm -f *.pdf *.png *~ .*~
	rm -rf __pycache__/ .mypy_cache/
