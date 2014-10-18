#!/bin/bash
if [ $# -ne 1 ]
    then
    echo "Usage: "$0" [Test|Store]"
    exit 1
fi

PROGRAM=../DOpE-PDE-InstatPDE-Example4-1d-1d

if [ -f dope.log ]
then
	rm dope.log
fi

if [ $1 == "Test" ]
then
    if [ -f test.dlog ]
    then
	echo "Running Program ${PROGRAM} test.prm"
	(${PROGRAM} test.prm 2>&1) > /dev/null
	echo "Comparing Results:"
	(diff dope.log test.dlog 2>&1) > /dev/null
	if [ $? -eq 0 ]
	then
	    echo "No differences found."
	    rm dope.log
	    exit 0
	else
	    echo "There where discrepancies in the Output."
	    diff dope.log test.dlog
	    exit 1
	fi
    else
	echo "No File test.dlog found for comparisson. Run '"$0" Store' to create one."
	exit 1
    fi
else
    if [ $1 == "Store" ]
    then
	echo "Running Program ${PROGRAM} test.prm"
	(${PROGRAM} test.prm 2>&1) > /dev/null
	echo "Run completed. Cleaning up ..."
	mv dope.log test.dlog	
	exit 0;
    else
	echo "Unknown Option: "$1
	exit 1
    fi
fi
    
