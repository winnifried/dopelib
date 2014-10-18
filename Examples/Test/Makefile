# $Id: Makefile,v 1.5 2012/02/14 07:58:05 wollner Exp $

# The large projects Makefile looks much like the one for small
# projects. Basically, only the following six parameters need to be
# set by you:

# The first denotes the dimension for which the program is to be
# compiled:
deal_II_dimension = 2
dope_dimension  = 2

# the DOpE toplevel directory:
#DOpE = ../../

DOpE = ../../
BINDIR = ./
# The second tells us the name of the executable. It is prefixed by
# `lib/' to designate its destination directory. Note that the program
# name depends on the dimension, so you can keep copies for the
# different dimensions around:
target   = $(BINDIR)/DOpE-test-$(dope_dimension)d-$(deal_II_dimension)d

# The `debug-mode' variable works as in the small projects Makefile:
debug-mode = off

# And so does the following variable. You will have to set it to
# something more reasonable, of course.
clean-up-files = ...

# Finally, here is a variable which tells the `run' rule which
# parameters to pass to the executable. Usually, this will be the name
# of an input file.

#run-parameters  = parameter-file.prm


# Now, this is the last variable you need to set, namely the path to
# the deal.II toplevel directory:
#D = ../../deal.II

D = $(HOME)/deal.II
#D = ~home/deal.II

#
#
# Usually, you will not need to change anything beyond this point.
#
#
# This tells `make' where to find the global settings and rules:
include $D/common/Make.global_options


# First get a list of files belonging to the project. Include files
# are expected in `include/', while implementation files are expected
# in `source/'. Object files are placed into `lib/[123]d', using the
# same base name as the `.cc' file.
cc-files    = $(shell echo *.cc)
o-files     = $(cc-files:%.cc=%.$(OBJEXT))
go-files    = $(cc-files:%.cc=%.g.$(OBJEXT))
h-files     = $(wildcard *.h)

ifeq ($(DEAL_II_MAJOR),6)
lib-h-files = $(shell echo $D/base/include/base/*.h \
		 $D/lac/include/lac/*.h             \
		 $D/deal.II/include/*/*.h	    \
	  	 $(DOpE)DOpEsrc/*/*.h)
libs.g   = $(lib-deal2-$(deal_II_dimension)d.g) \
	   $(lib-lac.g)                         \
           $(lib-base.g)			\
	   $(lib-dope-$(dope_dimension)d-$(deal_II_dimension)d.g)
libs.o   = $(lib-deal2-$(deal_II_dimension)d.o) \
	   $(lib-lac.o)                         \
           $(lib-base.o)			\
	   $(lib-dope-$(dope_dimension)d-$(deal_II_dimension)d.o)
else
lib-h-files = $(shell echo $D/include/deal.II/*/*.h	    \
	  	 $(DOpE)DOpEsrc/*/*.h)	
libs.g   = $(lib-deal2.g) \
	   $(lib-dope-$(dope_dimension)d-$(deal_II_dimension)d.g)
libs.o   = $(lib-deal2.o) \
	   $(lib-dope-$(dope_dimension)d-$(deal_II_dimension)d.o)
endif

# Include lib-files from DOpE toolkit
lib-dope-$(dope_dimension)d-$(deal_II_dimension)d.g = $(DOpE)DOpEsrc/lib/$(dope_dimension)d/$(deal_II_dimension)d/libdope.g.a
lib-dope-$(dope_dimension)d-$(deal_II_dimension)d.o = $(DOpE)DOpEsrc/lib/$(dope_dimension)d/$(deal_II_dimension)d/libdope.a

# Define a nifty string to indicate in the output of the compile
# commands whether the program is compiled in multithread mode or not:
ifneq ($(with-multithreading),no)
  MT = MT
else
  MT = ==
endif

CXXFLAGS.g += -std=gnu++0x
CXXFLAGS.o += -std=gnu++0x

# Now use the information from above to define the set of libraries to
# link with and the flags to be passed to the compiler:
ifeq ($(debug-mode),on)
  libraries = $(go-files) $(libs.g) 
  flags     = $(CXXFLAGS.g)
else
  libraries = $(o-files) $(libs.o)
  flags     = $(CXXFLAGS.o)
endif


# Then augment the compiler flags by a specification of the dimension
# for which the program shall be compiled:
flags += -Ddeal_II_dimension=$(deal_II_dimension) -Ddope_dimension=$(dope_dimension)

LIBS += -I$(DOpE)/DOpEsrc/include/ -I$(DOpE)/DOpEsrc/templates -I$(DOpE)/DOpEsrc/wrapper -I$(DOpE)/DOpEsrc/basic -I$(DOpE)/DOpEsrc/problemdata -I$(DOpE)/DOpEsrc/interfaces -I$(DOpE)/DOpEsrc/reducedproblems -I$(DOpE)/DOpEsrc/container

# The following two rules define how to compile C++ files into object
# files:
%.g.$(OBJEXT) :
	@echo =====waves=======$(dope_dimension)d==$(deal_II_dimension)d====debug=====$(MT)== $(<F)
	@$(CXX) $(flags) $(LIBS) -c $< -o $@
%.$(OBJEXT) :
	@echo =====waves=======$(dope_dimension)d==$(deal_II_dimension)d====optimized=$(MT)== $(<F)
	@$(CXX) $(flags) $(LIBS) -c $< -o $@


# Next define how to link the executable
$(target)$(EXEEXT) : $(libraries) Makefile
	@echo =====waves=======$(dope_dimension)d==$(deal_II_dimension)d==============$(MT)== Linking $(@F)
	@$(CXX) -o $@ $(libraries) $(LIBS) $(LDFLAGS)



# Rule how to run the program
run: $(target)$(EXEEXT)
	./$(target)$(EXEEXT) $(run-parameters)


# Rule how to clean up. This is split into several different rules to
# allow for parallel execution of commands:
clean: clean-lib clean-data
	-rm -f *~ */*~ */*/*~ Makefile.dep

clean-lib:
	-rm -f *.$(OBJEXT) *.g.$(OBJEXT) $(target)$(EXEEXT) TAGS

clean-data:
	-rm -f $(clean-up-files)


# Again tell `make' which rules are not meant to produce files:
.PHONY: clean clean-data clean-lib run



# Finally produce the list of dependencies. Note that this time, the
# object files end up in directories of their own, so we have to
# modify the output a bit. The file with the dependencies is put into
# `lib/'.
Makefile.dep: $(cc-files) $(h-files) $(lib-h-files) Makefile
	@echo =====waves=======$(dope_dimension)d==$(deal_II_dimension)d================== Remaking $@
	@$D/common/scripts/make_dependencies $(INCLUDE) -B. $(cc-files) \
	 | $(PERL) -p -e 's!^(.*):!$$1:!g;' \
		> $@

include Makefile.dep