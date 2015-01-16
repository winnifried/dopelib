help:
	@echo "========================================================================"
	@echo "=              Makefile for the DOpE libraries                         ="
	@echo "========================================================================"
	@echo "=                                                                      ="
	@echo "= The following targets exist:                                         ="
	@echo "=    all      : Create all DOpE libraries                              ="
	@echo "=    c-all    : Create all DOpE libraries using the cmake system       ="
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
c-all:
	./autocmake.sh configure
	cd autobuild/release; $(MAKE)
	cd autobuild/debug; $(MAKE)
debug:
	cd source ; $(MAKE) debug

opt:
	cd source ; $(MAKE) optimized

pdf-doc:	
	cd ../Examples ; $(MAKE) doc

html-doc:
	cd ../doxygen ; $(MAKE) all
clean:
	./autocmake.sh clean
	rm -rf *~ TAGS 
	cd source ; $(MAKE) clean
	rm lib/?d/?d/*.a

distclean:
	./autocmake.sh distclean
	rm -rf *~ TAGS 
	cd source ; $(MAKE) veryclean
	cd ../doxygen ; $(MAKE) clean
	cd ../Examples ; $(MAKE) distclean

TAGS:
	etags include/*.h source/*.cc source/*.imp 

.PHONY: TAGS
