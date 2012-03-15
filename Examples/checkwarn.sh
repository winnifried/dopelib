#!/bin/bash

CURRENT=`pwd`;
if [ -f comp.txt ]
then
    rm comp.txt;
fi
touch comp.txt
for bd in OPT PDE
do echo "Trying "$bd 
    if [ -d $bd ]
    then
	cd $bd
	for sd in StatPDE InstatPDE
	do echo "Trying "$bd"/"$sd
	    if [ -d $sd ]
	    then
		cd $sd
		for i in `ls -d Example*`
		do cd $i
		    echo "Cleaning "$i
		    (make clean 2>&1 ) > /dev/null
		    echo "Making ..."
		    echo "In Directory: "$bd"/"$sd"/"$i >> $CURRENT/comp.txt
		    (make debug-mode=on 2>&1) >>  $CURRENT/comp.txt
		    (make clean 2>&1 ) > /dev/null
		    (make 2>&1 ) > /dev/null
		    cd ..
		done
		cd ..
	    fi
	    wait
	done
	cd ..
    fi
done
(grep warning\\\|error\\\|Fehler\\\|Warnung comp.txt 2>&1) > /dev/null
if [ $? -eq 0 ]
then
    echo "There are warnings:"
    grep Directory\\\|warning\\\|error\\\|Fehler\\\|Warnung comp.txt
else
    echo "No warnings found"
    rm comp.txt
fi