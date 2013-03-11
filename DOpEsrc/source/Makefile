all: debug optimized
debug: libdope.g.a
optimized:  libdope.a

############################################################
# Directory with the DEAL distribution
############################################################
D = $(HOME)/deal.II/

############################################################
# DOpEsrc Directory 
############################################################
DOpE = ../
############################################################
# Source Files
############################################################

#Where files should go
OBJDIR = $(DOpE)/obj/
SRCDIR = $(DOpE)/source/ 

## create lists of file names
cc-files = $(shell echo *.cc)
nodim-cc-files = 

h-files = $(wildcard ../include/*.h) $(wildcard ../templates/*.h) $(wildcard ../wrapper/*.h) $(wildcard ../basic/*.h) $(wildcard ../problemdata/*.h) $(wildcard ../interfaces/*.h) $(wildcard ../container/*.h) $(wildcard ../opt_algorithms/*.h) $(wildcard ../tsschemes/*.h)

############################################################
# Include general settings for including DEAL libraries
############################################################

include $D/common/Make.global_options

CXXFLAGS += -std=gnu++0x -Wall -fvisibility-inlines-hidden

FLAGS = $(CXXFLAGS) -DGLIBCXX_DEBUG

############################################################
# Check the deal version
############################################################

DEAL_II_MAJOR_VERSION = $(DEAL_II_MAJOR)
DEAL_II_MINOR_VERSION = $(DEAL_II_MINOR)

############################################################
# Targets for compilation
############################################################
#first dope then deal dimension!
libdope.g.a : lib01d-deb lib11d-deb lib02d-deb lib22d-deb lib03d-deb lib33d-deb
#libdope.g.a : $(DOpE)/lib/1d/1d/libdope.g.a $(DOpE)/lib/2d/2d/libdope.g.a $(DOpE)/lib/3d/3d/libdope.g.a

libdope.a : lib01d lib11d lib02d lib22d lib03d lib33d
#libdope.a : $(DOpE)/lib/1d/1d/libdope.a $(DOpE)/lib/2d/2d/libdope.a $(DOpE)/lib/3d/3d/libdope.a

lib01d : $(DOpE)/lib/1d/1d/libdope.a
lib01d-deb : $(DOpE)/lib/1d/1d/libdope.g.a
lib11d : $(DOpE)/lib/1d/1d/libdope.a
lib11d-deb : $(DOpE)/lib/1d/1d/libdope.g.a
lib02d : $(DOpE)/lib/0d/2d/libdope.a
lib02d-deb : $(DOpE)/lib/0d/2d/libdope.g.a
lib22d : $(DOpE)/lib/2d/2d/libdope.a
lib22d-deb : $(DOpE)/lib/2d/2d/libdope.g.a
lib03d : $(DOpE)/lib/0d/3d/libdope.a
lib03d-deb : $(DOpE)/lib/0d/3d/libdope.g.a
lib33d : $(DOpE)/lib/3d/3d/libdope.a
lib33d-deb : $(DOpE)/lib/3d/3d/libdope.g.a

$(OBJDIR)%.0d.1d.g.o : %.cc $(h-files)
	@echo =====debug========== $@
	@$(CXX) $(CXXFLAGS.g) $(FLAGS) -I../.. -Ddeal_II_dimension=1  -Ddope_dimension=0 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@
$(OBJDIR)%.0d.1d.o : %.cc $(h-files)
	@echo =====optimized====== $@
	@$(CXX) $(CXXFLAGS.o) $(FLAGS) -I../.. -Ddeal_II_dimension=1  -Ddope_dimension=0 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@

$(OBJDIR)%.1d.1d.g.o : %.cc $(h-files)
	@echo =====debug========== $@
	@$(CXX) $(CXXFLAGS.g) $(FLAGS) -I../.. -Ddeal_II_dimension=1  -Ddope_dimension=1 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@
$(OBJDIR)%.1d.1d.o : %.cc $(h-files)
	@echo =====optimized====== $@
	@$(CXX) $(CXXFLAGS.o) $(FLAGS) -I../.. -Ddeal_II_dimension=1  -Ddope_dimension=1 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@

$(OBJDIR)%.0d.2d.g.o : %.cc $(h-files)
	@echo =====debug========== $@
	@$(CXX) $(CXXFLAGS.g) $(FLAGS) -I../.. -Ddeal_II_dimension=2  -Ddope_dimension=0 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@
$(OBJDIR)%.0d.2d.o : %.cc $(h-files)
	@echo =====optimized====== $@
	@$(CXX) $(CXXFLAGS.o) $(FLAGS) -I../.. -Ddeal_II_dimension=2  -Ddope_dimension=0 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@

$(OBJDIR)%.2d.2d.g.o : %.cc $(h-files)
	@echo =====debug========== $@
	@$(CXX) $(CXXFLAGS.g) $(FLAGS) -I../.. -Ddeal_II_dimension=2  -Ddope_dimension=2 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@
$(OBJDIR)%.2d.2d.o : %.cc $(h-files)
	@echo =====optimized====== $@
	@$(CXX) $(CXXFLAGS.o) $(FLAGS) -I../.. -Ddeal_II_dimension=2  -Ddope_dimension=2 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@

$(OBJDIR)%.0d.3d.g.o : %.cc $(h-files)
	@echo =====debug========== $@
	@$(CXX) $(CXXFLAGS.g) $(FLAGS) -I../.. -Ddeal_II_dimension=3  -Ddope_dimension=0 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@
$(OBJDIR)%.0d.3d.o : %.cc $(h-files)
	@echo =====optimized====== $@
	@$(CXX) $(CXXFLAGS.o) $(FLAGS) -I../.. -Ddeal_II_dimension=3  -Ddope_dimension=0 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@

$(OBJDIR)%.3d.3d.g.o : %.cc $(h-files)
	@echo =====debug========== $@
	@$(CXX) $(CXXFLAGS.g) $(FLAGS) -I../.. -Ddeal_II_dimension=3  -Ddope_dimension=3 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@
$(OBJDIR)%.3d.3d.o : %.cc $(h-files)
	@echo =====optimized====== $@
	@$(CXX) $(CXXFLAGS.o) $(FLAGS) -I../.. -Ddeal_II_dimension=3  -Ddope_dimension=3 -DDEAL_II_MAJOR_VERSION=$(DEAL_II_MAJOR_VERSION) -DDEAL_II_MINOR_VERSION=$(DEAL_II_MINOR_VERSION) -c $< -o $@

o0-1-files = $(nodim-cc-files:.cc=.o) $(cc-files:%.cc=$(OBJDIR)%.0d.1d.o)
o1-1-files = $(nodim-cc-files:.cc=.o) $(cc-files:%.cc=$(OBJDIR)%.1d.1d.o)
o0-2-files = $(nodim-cc-files:.cc=.o) $(cc-files:%.cc=$(OBJDIR)%.0d.2d.o)
o2-2-files = $(nodim-cc-files:.cc=.o) $(cc-files:%.cc=$(OBJDIR)%.2d.2d.o)
o0-3-files = $(nodim-cc-files:.cc=.o) $(cc-files:%.cc=$(OBJDIR)%.0d.3d.o)
o3-3-files = $(nodim-cc-files:.cc=.o) $(cc-files:%.cc=$(OBJDIR)%.3d.3d.o)
g0-1-files = $(nodim-cc-files:.cc=.g.o) $(cc-files:%.cc=$(OBJDIR)%.0d.1d.g.o)
g1-1-files = $(nodim-cc-files:.cc=.g.o) $(cc-files:%.cc=$(OBJDIR)%.1d.1d.g.o)
g0-2-files = $(nodim-cc-files:.cc=.g.o) $(cc-files:%.cc=$(OBJDIR)%.0d.2d.g.o)
g2-2-files = $(nodim-cc-files:.cc=.g.o) $(cc-files:%.cc=$(OBJDIR)%.2d.2d.g.o)
g0-3-files = $(nodim-cc-files:.cc=.g.o) $(cc-files:%.cc=$(OBJDIR)%.0d.3d.g.o)
g3-3-files = $(nodim-cc-files:.cc=.g.o) $(cc-files:%.cc=$(OBJDIR)%.3d.3d.g.o)

$(DOpE)/lib/0d/1d/libdope.g.a : $(g0-1-files)
	@ar  ruv $@ $^
$(DOpE)/lib/1d/1d/libdope.g.a : $(g1-1-files)
	@ar  ruv $@ $^
$(DOpE)/lib/0d/2d/libdope.g.a : $(g0-2-files)
	@ar  ruv $@ $^
$(DOpE)/lib/2d/2d/libdope.g.a : $(g2-2-files)
	@ar  ruv $@ $^
$(DOpE)/lib/0d/3d/libdope.g.a : $(g0-3-files)
	@ar  ruv $@ $^
$(DOpE)/lib/3d/3d/libdope.g.a : $(g3-3-files)
	@ar  ruv $@ $^
$(DOpE)/lib/0d/1d/libdope.a : $(o0-1-files)
	@ar  ruv $@ $^
$(DOpE)/lib/1d/1d/libdope.a : $(o1-1-files)
	@ar  ruv $@ $^
$(DOpE)/lib/0d/2d/libdope.a : $(o0-2-files)
	@ar  ruv $@ $^
$(DOpE)/lib/2d/2d/libdope.a : $(o2-2-files)
	@ar  ruv $@ $^
$(DOpE)/lib/0d/3d/libdope.a : $(o0-3-files)
	@ar  ruv $@ $^
$(DOpE)/lib/3d/3d/libdope.a : $(o3-3-files)
	@ar  ruv $@ $^

############################################################
# Cleanup targets
############################################################

clean:
	rm -f $(OBJDIR)*.o $(OBJDIR)*.go

veryclean: clean
	rm -f *.a *.inp *.gpl *.eps *.gnuplot Makefile.depend
	rm ../lib/?d/?d/*.a

INCLUDE += -I../include -I../templates -I../wrapper -I../basic -I../problemdata -I../interfaces -I../reducedproblems -I../container -I../opt_algorithms -I../tsschemes

all-cc-files = $(shell echo *.cc)

Makefile.depend: $(all-cc-files)  $(h-files) Makefile
	@echo =====Dependencies== Makefile.depend
	@$D/common/scripts/make_dependencies $(INCLUDE) -I../.. *.cc $(h-files) > $@
	@perl -pi~ -e 's/(^[^.]+)\.o:/\1.o \1.g.o \1.2d.o \1.3d.o \1.2d.g.o \1.3d.g.o:/;' $@

include Makefile.depend
