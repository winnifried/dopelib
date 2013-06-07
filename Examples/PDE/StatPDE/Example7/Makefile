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
target   = $(BINDIR)/DOpE-PDE-StatPDE-Example7-$(dope_dimension)d-$(deal_II_dimension)d

# The `debug-mode' variable works as in the small projects Makefile:
debug-mode = off

# And so does the following variable. You will have to set it to
# something more reasonable, of course.
clean-up-files = ...

#Read the default values for all examples
include $(DOpE)/Examples/Make.global_options

