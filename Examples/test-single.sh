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
	    #Which Version of Deal.II are we using?
	    if [ ! -f dope.log ]
	    then
		echo "No logfile written"
		exit 1
	    fi

	    for i in `ls test*.dlog`
	    do
		log=$i
		#We don't compare the header (first seven lines of the log file)
		(diff <(tail -n +8 dope.log) <(tail -n +8 ${log}) 2>&1) > /dev/null
		if [ $? -eq 0 ]
		then
		    echo "No differences found to log "$log
		    rm dope.log
		    if [ -d Mesh0 ] 
		    then
			rm -r Mesh?/
		    fi
		    if [ -f grid.eps ]
		    then 
			rm grid.eps
		    fi
		    if [ -d tmp_state ]
		    then
			rm -r tmp_*
		    fi
		    exit 0
		else
		    echo "There where discrepancies in the Output compared to "$log		    
		fi
	    done
	    #When we get here, all log files had discrepancies 
	    
	    if [ -d Mesh0 ] 
	    then
		rm -r Mesh?/
	    fi
	    if [ -f grid.eps ]
	    then 
		rm grid.eps
	    fi
	    if [ -d tmp_state ]
	    then
		rm -r tmp_*
		fi
	    exit 1
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
	    if [ -d tmp_state ]
	    then
		rm -r tmp_*
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
