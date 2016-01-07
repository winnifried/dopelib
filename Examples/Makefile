DIRECTORIES = $(shell echo OPT/StatPDE/Example* \
				OPT/InstatPDE/Example* \
				PDE/StatPDE/Example* \
				PDE/InstatPDE/Example* )

.PHONY: $(DIRECTORIES) all

help:
	@echo "==========================================================================="
	@echo "=              Makefile for the DOpE documentation                        ="
	@echo "==========================================================================="
	@echo "=                                                                         ="
	@echo "= The following targets exist:                                            ="
	@echo "=    all       :  Make all examples (in optimized mode)                   ="
	@echo "=    c-all     :  Make all examples using cmake                           ="
	@echo "=    allignore :  Make all examples (in optimized mode), ignore errors.   ="
	@echo "=    clean     :  Cleaning up all examples                                ="
	@echo "=    tests     :  Run all test param data.                                ="
	@echo "=    cat       :  Run clean, run all, run tests (combine these commands)  ="
	@echo "=    c-cat     :  Run clean, run c-all, run tests (combine these commands)="
	@echo "=    catignore :  Same as cat, but ignores compiler errors.               ="	
	@echo "=    pdf-doc   :  Create documentation in pdf file format via latexmk     ="
	@echo "=    doc       :  Create documentation in pdf file format via latexmk     ="
	@echo "=    distclean :  Cleaning up, including documentation                    ="
	@echo "=    warncheck :  Checks whether all Examples compile without warnings    ="
	@echo "==========================================================================="

$(DIRECTORIES):
	cd $@; $(MAKE) -s $(MAKECMDGOALS)

all c-all clean veryclean:	$(DIRECTORIES)

allignore:
	$(MAKE) -i all;

tests:  
	./testall.sh

cat:	
	$(MAKE) clean; $(MAKE) all ; ./testall.sh
c-cat:	
	$(MAKE) clean; $(MAKE) c-all ; ./testall.sh

catignore:	
	$(MAKE) clean; $(MAKE) -i all ; ./testall.sh

pdf-doc doc:
	latexmk -pdf description.tex
distclean:
	rm -rf *~ 
	$(MAKE) veryclean 
	rm -f description.aux description.aux.bak description.log description.pdf description.toc description.bbl description.blg description.fdb_latexmk description.fls description.idx description.ilg description.ind 
	rm -f Documentation/*.aux Documentation/*.aux.bak 

warncheck:
	./checkwarn.sh
