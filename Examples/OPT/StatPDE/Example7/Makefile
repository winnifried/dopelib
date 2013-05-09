# The first denotes the dimension for which the program is to be
# compiled:
deal_II_dimension = 2
dope_dimension  = 2

# the DOpE toplevel directory:
DOpE = ../../../../
BINDIR = $(DOpE)/bin/

# The second tells us the name of the executable. It is prefixed by
# `lib/' to designate its destination directory. Note that the program
# name depends on the dimension, so you can keep copies for the
# different dimensions around:
target   = $(BINDIR)/DOpE-OPT-StatPDE-Example7-$(dope_dimension)d-$(deal_II_dimension)d

# The `debug-mode' variable works as in the small projects Makefile:
#debug-mode = on
debug-mode = off

# And so does the following variable. You will have to set it to
# something more reasonable, of course.
clean-up-files = ...

_LOCAL_thirdpartylibs = $(shell echo $(DOpE)/ThirdPartyLibs/*)

ifneq (,$(findstring snopt, $(_LOCAL_thirdpartylibs)))
DOPE_FLAGS += -DWITH_SNOPT -fPIC
SNOPT = $(DOpE)/ThirdPartyLibs/snopt

SNOPT_F2CINC = $(SNOPT)/include/
SNOPT_F2CLIB = $(SNOPT)/lib/

SNOPT_LIBDIR = $(SNOPT)/lib/

SNOPT_LIBS    = -lsnopt_cpp -lsnopt -lsnprint -lblas
SNOPT_AR      = libsnopt_cpp libsnopt libsnprint libblas
SNOPT_AR_LIBS = $(SNOPT_AR:%=$(SNOPT_LIBDIR)/%.a)

#Additional library header files 
DOPE_LIBS += -I$(SNOPT)/cppsrc -I$(SNOPT_F2CINC) 

#additional linker information
DOPE_LDFLAGS += -L$(SNOPT_LIBDIR) $(SNOPT_AR_LIBS) $(SNOPT_F2CLIB)/libf2c.a 
#use fortran libs
DOPE_LDFLAGS += -lgfortranbegin -lgfortran

#endof With SNOPT	
endif

ifneq (,$(findstring ipopt, $(_LOCAL_thirdpartylibs)))
DOPE_FLAGS += -DWITH_IPOPT 
IPOPT = $(DOpE)/ThirdPartyLibs/ipopt/

IPOPT_INC = $(IPOPT)/include/coin/
IPOPT_LIB = $(IPOPT)/lib64/


#Additional library header files 
DOPE_LIBS += -I$(IPOPT_INC)  

IPOPT_LDFLAGS = -lipopt

#Check additionally installed libs
_LOCAL_ipopt_add_libs = $(shell echo $(IPOPT_LIB)/*)
ifneq (,$(findstring coinhsl, $(_LOCAL_ipopt_add_libs)))
IPOPT_LDFLAGS += -lcoinhsl		
endif
ifneq (,$(findstring coinmetis, $(_LOCAL_ipopt_add_libs)))
IPOPT_LDFLAGS += -lcoinmetis		
endif
ifneq (,$(findstring coinmumps, $(_LOCAL_ipopt_add_libs)))
IPOPT_LDFLAGS += -lcoinmumps		
endif
ifneq (,$(findstring coinlapack, $(_LOCAL_ipopt_add_libs)))
IPOPT_LDFLAGS += -lcoinlapack		
endif

#additional linker information
DOPE_LDFLAGS += -L$(IPOPT_LIB) $(IPOPT_LDFLAGS)

#endof With IPOPT	
endif

#Read the default values for all examples
include $(DOpE)/Examples/Make.global_options

