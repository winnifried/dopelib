DIRECTORIES = $(shell echo OPT/StatPDE/Example* \
				OPT/InstatPDE/Example* \
				PDE/StatPDE/Example* \
				PDE/InstatPDE/Example* )
PROCS ?= 1

.PHONY: $(DIRECTORIES) all

help:
	@echo "==========================================================================="
	@echo "=              Makefile for the DOpE documentation                        ="
	@echo "==========================================================================="
	@echo "=                                                                         ="
	@echo "= The following targets exist:                                            ="
	@echo "=    c-all     :  Make all examples using cmake                           ="
	@echo "=    c-all-deb :  Make all examples using cmake in Debug mode             ="
	@echo "=                 WARNING: Both do not reconfigure if previous cmake      ="
	@echo "=                    config used a different build type. Consider using   ="
	@echo "=                    make distclean before!                               ="
	@echo "=    clean     :  Cleaning up all examples                                ="
	@echo "=    tests     :  Run all test param data.                                ="
	@echo "=                 To run tests in parallel add PROCS=[number]             ="
	@echo "=                 to the call, i.e., make tests PROCS=2                   ="
	@echo "=    time      :  Run all test param data and report timings.             ="
	@echo "=                 To run tests in parallel add PROCS=[number]             ="
	@echo "=                 to the call, i.e., make tests PROCS=2                   ="
	@echo "=    c-cat     :  Run clean, run c-all, run tests (combine these commands)="
	@echo "=    doc       :  Create documentation in pdf file format via latexmk     ="
	@echo "=    distclean :  Cleaning up, including documentation                    ="
	@echo "=    warncheck :  Checks whether all Examples compile without warnings    ="
	@echo "==========================================================================="

$(DIRECTORIES):
	cd $@; $(MAKE) -s $(MAKECMDGOALS)

c-all c-all-deb clean veryclean:	$(DIRECTORIES)

tests:  
	./testall.sh -j$(PROCS)

time:  
	./testtime.sh -j$(PROCS)

c-cat:	
	$(MAKE) clean; $(MAKE) c-all ; ./testall.sh

doc:
	latexmk -pdf description.tex
distclean:
	rm -rf *~ 
	$(MAKE) veryclean 
	rm -f description.aux description.aux.bak description.log description.pdf description.toc description.bbl description.blg description.fdb_latexmk description.fls description.idx description.ilg description.ind description.synctex.gz
	rm -f Documentation/*.aux Documentation/*.aux.bak 

warncheck:
	./checkwarn.sh
