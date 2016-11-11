#!/bin/bash

if [ $# -ne 1 ]
    then
    echo "Usage: "$0" [configure|clean]"
    exit 1
fi

if [ $1 == "configure" ]
then
    if [ -d autobuild ]
    then
    #nothing to do, files already present
	exit 0
    else
	mkdir autobuild
	cd autobuild
	cmake -DCMAKE_BUILD_TYPE=Release ../
	cd ../
    #done with configure
	exit 0
    fi
else 
    if [ $1 == "distclean" ]
    then
	if [ -d autobuild ]
	then
      #remove files 
      	    cd autobuild
      	    make clean;
      	    cd ../
	    rm -r autobuild
	fi
	exit 0
    else
	if [ $1 == "clean" ]
	then
	    if [ -d autobuild ]
	    then
		cd autobuild
		make clean;
		cd ../
	    fi
	    exit 0
	fi
    fi
fi
echo "Unknown option "$1
exit 1
