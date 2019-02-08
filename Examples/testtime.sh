#!/bin/bash
failed=0
#The individual Tests:
function RUN_TEST {
    if [ -d $1 ]
    then
	cd $1
	echo -n "In "$2"/"$1" Test: " 
	if [ -d Test ]
	then
	    cd Test
	    if [ -f test.sh ]
	    then
		((time (./test.sh Test 2>&1 ) > /dev/null ) 2>&1 )> time.txt
		if [ $? -eq 0 ]
		then
		    echo -en '\E[32;40m'"succeeded!"
		    tput sgr0
		    echo -n " "
		    sed 's/s$/s\t/g' time.txt | sed 's/real\t/r: /g' | sed 's/user\t/u: /g' | sed 's/sys\t/s: /g' | tr -d '\n'
		    echo
		else
		    echo -en '\E[31;40m'"failed!"
		    tput sgr0
		    echo -n " "
		    sed 's/s$/s\t/g' time.txt | sed 's/real\t/r: /g' | sed 's/user\t/u: /g' | sed 's/sys\t/s: /g' | tr -d '\n'
		    echo 	
		    failed=$((${failed} + 1))
		fi
	    else
		echo -en '\E[31;40m'"failed!"
		tput sgr0
		echo 
		failed=$((${failed} + 1))
	    fi
	    cd ..
	else
	    echo -en "'\E[31;40m'failed!"
	    tput sgr0
	    echo
	    failed=$((${failed} + 1))
	fi
	cd ..
    else
	echo "Could not find directory :"$1
    fi
}

export -f RUN_TEST

while getopts 'j:' flag; do
    case "${flag}" in
	j) n_procs="${OPTARG}" ;;
	*) echo "Unknown option ${flag}."
	   echo "Run 'compile-tests.sh -j[nprocs]'."
	   exit 1 ;;
    esac
done

echo "Running tests with ${n_procs} parallel processes."

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
		if [ $n_procs == 1 ]
		then
		    for i in `ls -d Example*`
		    do
			RUN_TEST $i $bd"/"$sd
		    done
		else
		    #-k option to keep the output in the same order as in the sequential case
		    parallel -k --no-notice RUN_TEST ::: "`ls -d Example*`" ::: "$bd/$sd"
		fi
		cd ..
	    fi
	    wait
	done
	cd ..
    fi
done
exit $failed
