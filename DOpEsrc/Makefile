help:
	@echo "========================================================================"
	@echo "=              Makefile for the DOpE libraries                         ="
	@echo "========================================================================"
	@echo "=                                                                      ="
	@echo "= The following targets exist:                                         ="
	@echo "=    c-all    : Create all DOpE libraries using the cmake system       ="
	@echo "=    pdf-doc  : Make documentation of DOpE examples in pdf file format ="
	@echo "=    html-doc : Make html documentation of all DOpE library files      ="
	@echo "=    clean    : Remove all object files                                ="
	@echo "=    distclean: Remove all generated files (includes doc and examples) ="
	@echo "========================================================================"

c-all:
	./autocmake.sh configure
	cd autobuild/release; $(MAKE) 
	cd autobuild/debug; $(MAKE)

pdf-doc:	
	cd ../Examples ; $(MAKE) doc

html-doc:
	cd ../doxygen ; $(MAKE) all
clean:
	./autocmake.sh clean

distclean:
	./autocmake.sh distclean
	cd ../doxygen ; $(MAKE) clean
	cd ../Examples ; $(MAKE) distclean


