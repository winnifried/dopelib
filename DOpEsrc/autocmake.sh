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
	mkdir release
	mkdir debug
	cd release
	cmake -D CMAKE_BUILD_TYPE=release ../../
	cd ..
	cd debug
	cmake -D CMAKE_BUILD_TYPE=debug ../../
	cd ../..
    #done with configure
	exit 0
    fi
else 
    if [ $1 == "distclean" ]
    then
	if [ -d autobuild ]
	then
      #remove files 
	    rm -r autobuild
	fi
	exit 0
    else
	if [ $1 == "clean" ]
	then
	    if [ -d autobuild ]
	    then
		cd autobuild/release
		make clean;
		cd ../debug
		make clean
		cd ../..
	    fi
	fi
    fi
fi
echo "Unknown option "$1
exit 1