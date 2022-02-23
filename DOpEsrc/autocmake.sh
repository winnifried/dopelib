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
	cmake -D CMAKE_BUILD_TYPE=Release ../../
	cd ..
	cd debug
	cmake -D CMAKE_BUILD_TYPE=Debug ../../
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
	if [ -f ../lib/cmake/DOpElib/DOpElibConfig.cmake ]
	then
	    rm ../lib/cmake/DOpElib/DOpElibConfig.cmake
	fi
	for i in `seq 0 3`
	do
	    for j in `seq 3`
	    do
		if [ -f ../lib/${i}d/${j}d/libdope.a ]
		then
		    rm ../lib/${i}d/${j}d/libdope.a
		fi
		if [ -f ../lib/${i}d/${j}d/libdope.g.a ]
		then
		    rm ../lib/${i}d/${j}d/libdope.g.a
		fi
	    done
	done
    
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
	    exit 0
	fi
    fi
fi
echo "Unknown option "$1
exit 1
