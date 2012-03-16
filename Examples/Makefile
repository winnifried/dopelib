DIRECTORIES = $(shell echo OPT/StatPDE/Example* \
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
	@echo "=    clean     :  Cleaning up all examples                                ="
	@echo "=    tests     :  Run all test param data.                                ="
	@echo "=    cat       :  Run clean, run all, run tests (combine these commands)  ="	
	@echo "=    pdf-doc   :  Create documentation in pdf file format via latexmk     ="
	@echo "=    doc       :  Create documentation in pdf file format via latexmk     ="
	@echo "=    distclean :  Cleaning up, including documentation                    ="
	@echo "=    warncheck :  Checks whether all Examples compile without warnings    ="
	@echo "==========================================================================="

$(DIRECTORIES):
	cd $@; $(MAKE) -s $(MAKECMDGOALS)

all clean veryclean:	$(DIRECTORIES)

tests:  
	./testall.sh

cat:	
	$(MAKE) clean; $(MAKE) all ; ./testall.sh

pdf-doc doc:
	latexmk -pdf description.tex
distclean:
	rm -rf *~ 
	$(MAKE) veryclean 
	rm description.aux description.aux.bak description.log description.pdf description.toc
	rm Documentation/*.aux Documentation/*.aux.bak

warncheck:
	./checkwarn.sh
