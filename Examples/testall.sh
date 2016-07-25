#!/bin/bash
failed=0

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
		    echo -n "In "$bd"/"$sd"/"$i" Test: " 
		    if [ -d Test ]
		    then
			cd Test
			if [ -f test.sh ]
			then
			    (./test.sh Test 2>&1 ) > /dev/null
			    if [ $? -eq 0 ]
			    then
				echo -en '\E[32;40m'"succeeded!"
				tput sgr0
				echo
			    else
				echo -en '\E[31;40m'"failed!"
				tput sgr0
				echo 	
				failed=1
			    fi
			else
			    echo -en '\E[31;40m'"failed!"
			    tput sgr0
			    echo 
			    failed=1
			fi
			cd ..
		    else
			echo -en "'\E[31;40m'failed!"
			tput sgr0
			echo
			failed=1
		    fi
		    cd ..
		done
		cd ..
	    fi
	    wait
	done
	cd ..
    fi
done
exit $failed
