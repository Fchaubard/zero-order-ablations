# Build the paper:  make        -> main.pdf  (pdflatex, bibtex, pdflatex, pdflatex)
#                   make dummy  -> regenerate placeholder CSVs / tables
#                   make page   -> render page 1 to page-1.png for a quick look
#                   make clean  -> remove build products
PDFLATEX ?= /opt/homebrew/bin/pdflatex
BIBTEX   ?= /opt/homebrew/bin/bibtex
PDFTOPPM ?= /opt/homebrew/bin/pdftoppm
PYTHON   ?= python3
MAIN      = main
FLAGS     = -interaction=nonstopmode -halt-on-error

SOURCES = $(MAIN).tex refs.bib neurips_2025.sty \
          $(wildcard figures/*.tex) $(wildcard tables/*.tex) $(wildcard data/*.csv)

.PHONY: all dummy page clean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	@# the shared pgfplots legend (\ref{zoolegend}) can shift labels once more
	@for i in 1 2; do \
	  if grep -q "Rerun to get cross-references right" $(MAIN).log; then \
	    $(PDFLATEX) $(FLAGS) $(MAIN) > /dev/null; \
	  fi; \
	done
	@grep -q "Rerun to get cross-references right" $(MAIN).log && echo "WARNING: cross-references still unstable" || true

dummy:
	$(PYTHON) make_dummy_data.py

page: $(MAIN).pdf
	$(PDFTOPPM) -png -r 60 -f 1 -l 1 $(MAIN).pdf page

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg $(MAIN).toc \
	      $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz $(MAIN).pdf page-*.png
