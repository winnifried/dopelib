help:
	@echo "========================================================================"
	@echo "=              Makefile for the DOpE libraries                         ="
	@echo "========================================================================"
	@echo "=                                                                      ="
	@echo "= The following targets exist:                                         ="
	@echo "=    all      : Create all DOpE libraries                              ="
	@echo "=    debug    : Create DOpE libraries in debug version                 ="
	@echo "=    opt      : Create DOpE libraries in optimized version             ="
	@echo "=    pdf-doc  : Make documentation of DOpE examples in pdf file format ="
	@echo "=    html-doc : Make html documentation of all DOpE library files      ="
	@echo "=    clean    : Remove all object files                                ="
	@echo "=    distclean: Remove all generated files (includes documentation)    ="
	@echo "=    TAGS     : Create a TAGS file from include and source files       ="
	@echo "========================================================================"

all: 
	cd source ;  $(MAKE)

debug:
	cd source ; $(MAKE) debug

opt:
	cd source ; $(MAKE) optimized

pdf-doc:	
	cd ../Examples ; $(MAKE) doc

html-doc:
	cd ../doxygen ; $(MAKE) all
clean:
	rm -rf *~ TAGS 
	cd source ; $(MAKE) clean
	rm lib/?d/?d/*.a

distclean:
	rm -rf *~ TAGS 
	cd source ; $(MAKE) clean
	rm lib/?d/?d/*.a
	cd ../doxygen ; $(MAKE) clean
	cd ../Examples ; $(MAKE) distclean

TAGS:
	etags include/*.h source/*.cc source/*.imp 

.PHONY: TAGS
