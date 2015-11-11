#!/bin/bash
if [ $# -ne 2 ]
    then
    echo "Usage: "$0" [Test|Store] [Executable]"
    exit 1
fi

if [ -f dope.log ]
then
	rm dope.log
fi

if [ $1 == "Test" ]
then
    if [ -f test.dlog ]
    then
	if [ -f $2 ]
	    then
	    echo "Running Program $2 test.prm"
	    ($2 test.prm 2>&1) > /dev/null
	    echo "Comparing Results:"
	    (diff dope.log test.dlog 2>&1) > /dev/null
	    if [ $? -eq 0 ]
	    then
		echo "No differences found."
		rm dope.log
		if [ -d Mesh0 ] 
		then
		    rm -r Mesh?/
		fi
		if [ -f grid.eps ]
		then 
		    rm grid.eps
		fi
		exit 0
	    else
		echo "There where discrepancies in the Output."
		diff dope.log test.dlog
		if [ -d Mesh0 ] 
		then
		    rm -r Mesh?/
		fi
		if [ -f grid.eps ]
		then 
		    rm grid.eps
		fi
		exit 1
	    fi
	else
	    echo "Executable '"$2" not found."
	    exit 1
	fi
    else
	echo "No File test.dlog found for comparisson. Run '"$0" Store' to create one."
	exit 1
    fi
else
    if [ $1 == "Store" ]
    then
	if [ -f $2 ]
	then
	    echo "Running Program $2 test.prm"
	    ($2 test.prm 2>&1) > /dev/null
	    echo "Run completed. Cleaning up ..."
	    mv dope.log test.dlog
	    if [ -d Mesh0 ] 
	    then
		rm -r Mesh?/
	    fi
	    if [ -f grid.eps ]
	    then 
		rm grid.eps
	    fi
	    exit 0;
	else
	    echo "Executable '"$2" not found."
	    exit 1
	fi
    else
	echo "Unknown Option: "$1
	exit 1
    fi
fi
