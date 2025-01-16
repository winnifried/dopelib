#!/bin/bash
#Test-script to check whether the selected version of DOpE 
#compiles correctly
#@TU-Darmstadt, run with 
#~/src/DOpE-svn/ThingsForMaintainer/Compile-Tests/compile-tests.sh -d . -s ~/DOpE/ThirdPartyLibs/snopt -i ~/DOpE/ThirdPartyLibs/ipopt

helpflag=0
dir=''
directory=''
snoptdir=''
ipoptdir=''
plain_deal_dir="dealii-candi-noaddons"
full_deal_dir="dealii-candi"
n_procs=1
base_errors=0
dealii_versions_plain="git 9.0.1 9.1.1 9.2.0 9.3.3 9.4.2 9.5.2 9.6.2"
#"8.5.0 8.4.1 8.3.0"
dealii_versions_full="git 9.0.1 9.1.1 9.2.0 9.3.3 9.4.2 9.5.2 9.6.2"
newest_dealii=9.6.2
trilinos_version=12-12-1
p4est_version=2.0
scalapack_version=2.0.2
candi_test=0
nocandi_test=0

##Set Environment
function SET_ENV_VAR {
    if [ $1 == "gcc" ]
    then
	export CXX=g++
	export CC=gcc
	export FC=gfortran
	export FF=gfortran
	echo ${CXX} ${CC} ${FC} ${FF}
	echo "Set Environment to g++/gcc"
    else
	if [ $1 == "mpi-gcc" ]
	then
	    if [ `uname` == "Darwin" ]
	    then
		export CXX=mpicxx-mpich-mp
		export CC=mpicc-mpich-mp
		export FC=mpif90-mpich-mp
		export FF=mpif77-mpich-mp
	    else
		export CXX=mpicxx.mpich
		export CC=mpicc.mpich
		export FC=mpif90.mpich
		export FF=mpif77.mpich
	    fi
	echo "Set Environment to MPI with g++/gcc"
	else
	    echo "Unknown case "$1 
	    exit 1
	fi
    fi
}

if (( $# == 0 ))
then
    echo "Missing options, run 'compile-tests.sh -h' for help."
    exit 1
fi

n_procs=1

while getopts 'hd:i:s:j:co' flag; do
    case "${flag}" in
	h) helpflag=1 ;;
	d) dir=`realpath ${OPTARG}` ;;
	s) snoptdir="${OPTARG}" ;;
	i) ipoptdir="${OPTARG}" ;;
	j) n_procs="${OPTARG}" ;;
	c) candi_test=1 ;;
	o) nocandi_test=1 ;;
	*) echo "Unknown option ${flag}."
	   echo "Run 'compile-tests.sh -h' for help."
	   exit 1 ;;
    esac
done

if (( ${helpflag} ))
then
    echo "Usage: $0 [-h] -d [pathname] -s [snopt-path] -i [ipopt-path] [-j [n_procs]] [-c] [-o]"
    echo "-h : Show this help"
    echo "-d : Path where the tests should be run"
    echo "-s : Path to snopt installation (Must exist but can be empty; then one test will fail)"
    echo "-i : Path to ipopt installation (Must exist but can be empty; then one test will fail)"
    echo "-j : Number of Processes to be used"
    echo "-c : Run tests using candi for the newest dealii version"
    echo "-o : Run tests for older dealii versions (without candi)"
    exit 0
fi

if [[ ! (${candi_test} -eq 1 || ${nocandi_test} -eq 1) ]]
then
    echo "No tests selected, please choose at least one of the '-c' or '-o' options!"
    exit 1
fi

while true; do
    read -p "Starting tests in ${dir} [y|n]?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit 0;;
	* ) echo "Please respond with 'y' or 'n'."
    esac
done

if [ ! -d ${dir} ]
then 
    echo "Directory '${dir}' not found"
    exit 1
else
    cd ${dir}
fi 

log=${dir}/dope-compile-log_`date +%Y%m%d`.txt

if [ -f ${log} ]
then
    echo "Found previous log file, deleting..."
    rm ${log}
fi
touch ${log}

if [ "${snoptdir}" == '' ]
then 
    echo "Warning: No SNOPT-Directory provided" | tee -a ${log}
    base_errors=$((${base_errors} + 1))
else
    if [ ! -d ${snoptdir} ]
    then 
	echo "SNOPT-Directory '${snoptdir}' not found" | tee -a ${log}
	exit 1
    fi 
fi
if [ "${ipoptdir}" == '' ]
then 
    echo "Warning: No IPOPT-Directory provided" | tee -a ${log}
    base_errors=$((${base_errors} + 1))
else
    if [ ! -d ${ipoptdir} ]
    then 
	echo "IPOPT-Directory '${ipoptdir}' not found" | tee -a ${log}
	exit 1
    fi 
fi

#Installation of DOpEdevel
export DOPE=${dir}/wollner_DOpEdevel
if [ -d ${DOPE} ]
then 
    echo "Found DOpEdevel, updating ..." | tee -a ${log}
    cd ${DOPE}
    git pull
else 
    echo "No DOpEdevel present, cloning ..." | tee -a ${log}
    git clone git@git.physnet.uni-hamburg.de:wwollner/dopedevel.git wollner_DOpEdevel
    if (( $? == 0 )) 
    then 
	if [ -d ${DOPE} ]
	then 
	    echo "Installation of DOpEdevel successful" | tee -a ${log}
	    cd ${DOPE}
	else 
	    echo "Installation of DOpEdevel failed" | tee -a ${log}
	    exit 1
	fi
    else 
	echo "Installation of DOpEdevel failed" | tee -a ${log}
	exit 1
    fi
fi

#Creating Thirdparty libs
#SNOPT
if [ "${snoptdir}" == '' ]
then 
    if [ -d ${DOPE}/ThirdPartyLibs/snopt ]
    then 
	echo "SNOPT present but no link provided" | tee -a ${log}
	exit 1
    fi 
else #Provided snopt link
    if [ -d ${DOPE}/ThirdPartyLibs/snopt ]
    then 
	echo "SNOPT present" | tee -a ${log}
    else
	echo "Creating symlink for SNOPT" | tee -a ${log}
	ln -s ${snoptdir} ${DOPE}/ThirdPartyLibs/snopt
    fi 
fi



#IPOPT
if [ "${ipoptdir}" == '' ]
then 
    if [ -d ${DOPE}/ThirdPartyLibs/ipopt ]
    then 
	echo "IPOPT present but no link provided" | tee -a ${log}
	exit 1
    fi 
else 
    if [ -d ${DOPE}/ThirdPartyLibs/ipopt ]
    then 
	echo "IPOPT present" | tee -a ${log}
    else
	echo "Creating symlink for IPOPT" | tee -a ${log}
	ln -s ${ipoptdir} ${DOPE}/ThirdPartyLibs/ipopt
    fi   
fi

if [ ${nocandi_test} -eq 1 ]
then
    echo "Running test of dealii (including past versions) without candi" | tee -a ${log}
    cd ${dir}
   
    #dealii plain
    for i in `echo ${dealii_versions_plain}`
    do
	if [[ ! -f install-dealii.sh ]]
	then
	    echo "No deal.ii install script found"
	    exit 1
	fi
	(./install-dealii.sh -d ${dir} -p -v $i -j${n_procs} 2>&1) | tee -a ${log}
	if [ ${PIPESTATUS[0]} -eq 0 ]
	then
	    echo "Example Build passed for dealii "${i}" plain" | tee -a ${log}
	else
	    echo "Example Build failed for dealii "${i}" plain" | tee -a ${log}
	fi
    done #End of deal ii plain install loop 
    #dealii full
    for i in `echo ${dealii_versions_full}`
    do
	if [[ ! -f install-dealii.sh ]]
	then
	    echo "No deal.ii install script found"
	    exit 1
	fi
	(./install-dealii.sh -d ${dir} -a -v $i -j${n_procs} 2>&1) | tee -a ${log}
	if [ ${PIPESTATUS[0]} -eq 0 ]
	then
	    echo "Example Build passed for dealii "${i}" full" | tee -a ${log}
	else
	    echo "Example Build failed for dealii "${i}" full" | tee -a ${log}
	fi
    done #End of deal ii full install loop 

    
    #Compile Tests
    #compiler loop.
    #clang doesn't work with mpi, so no real loop
    for cs in gcc 
    do 
	cd ${dir}
	
	if [ -d ${cs} ]
	then
	    echo "Subdirectory ${cs} present" | tee -a ${log}
	    directory=${dir}/${cs}
	else
	    echo "Subdirectory ${cs} not present, creating " | tee -a ${log}
	    mkdir ${cs}
	    if [ ! -d ${cs} ]
	    then 
		echo "Failed to create ${cs}" | tee -a ${log}
		exit 1
	    fi
	    directory=${dir}/${cs}
	fi
	cd ${directory}
	
	
	#Running tests
	echo "***************************************************************************" | tee -a ${log}
	echo "***************************************************************************" | tee -a ${log}
	echo "*            Running Tests for deal.ii without candi                      *" | tee -a ${log}
	echo "***************************************************************************" | tee -a ${log}
	echo "***************************************************************************" | tee -a ${log}
	for deal in plain full
	do
	    if [ "${deal}" == "plain" ]
	    then
		dealii_versions=${dealii_versions_plain}
	    else
		if [ "${deal}" == "full" ]
		then
		    dealii_versions=${dealii_versions_full}
		else
		    echo "Unkown flag: ${deal} " | tee -a ${log}
		    exit 1
		fi
	    fi
	    for dv in `echo ${dealii_versions}`
	    do
	    	echo "" | tee -a ${log}
		echo "***************************************************************************" | tee -a ${log}
		echo "Compile tests with ${deal} dealii v${dv}" | tee -a ${log}
	    
		if [ "${deal}" == "plain" ]
		then 
		    SET_ENV_VAR ${cs} 
		    echo "Compilers: ${CXX} ${CC} ${FC} ${FF}" | tee -a ${log}
		    export DEAL_II_DIR=${dir}/dealii-${dv}-${deal}-install
		else
		    if [ "${deal}" == "full" ]
		    then
			SET_ENV_VAR "mpi-"${cs} 
			echo "Compilers: ${CXX} ${CC} ${FC} ${FF}" | tee -a ${log}
			export DEAL_II_DIR=${dir}/dealii-${dv}-${deal}-install
    		    else
			echo "Unknown deal version: ${deal}" | tee -a ${log}
			exit 1
		    fi
		fi
		
		#Check if dealii dir is present
		if [ ! -d `echo ${DEAL_II_DIR}` ]
		then
		    echo "Can't find dealii directory ${DEAL_II_DIR}"
		else
		
		    cd ${DOPE}
		    cd DOpEsrc
		    echo "In `pwd`, cleaning ..." | tee -a ${log}
		    make distclean
		    echo "building..." | tee -a ${log}
		    (make c-all -j${n_procs} 2>&1) | tee ${dir}/${deal}_${dv}_${cs}.make
		    if [ ${PIPESTATUS[0]} -eq 0 ]
		    then
			echo "Build passed for "$cs" "${deal}" "${dv} | tee -a ${log}
		    else
			echo "Build failed for "$cs" "${deal}" "${dv} | tee -a ${log}
			break
		    fi
		    cd ${DOPE}
		    cd Examples 
		    echo "In `pwd`, building ..." | tee -a ${log}
		    (make c-all -j${n_procs} 2>&1) | tee ${dir}/${deal}_${dv}_${cs}.make-examples
		    if [ ${PIPESTATUS[0]} -eq 0 ]
		    then
			echo "Example Build passed for "$cs" "${deal}" "${dv} | tee -a ${log}
		    else
			echo "Example Build failed for "$cs" "${deal}" "${dv} | tee -a ${log}
			break
		    fi
		    echo "In `pwd`, testing ..." | tee -a ${log}
		    if [ ! -f ./testtime.sh ]
		    then 
			echo "Missing Testscript." | tee -a ${log}
			exit 1
		    fi
		    (./testtime.sh -j${n_procs} 2>&1) | tee -a ${dir}/${deal}_${dv}_${cs}.tests
		    n_errors=${PIPESTATUS[0]}
		    #Maybe some fail if no SNOPT or IPOPT are given
		    expected_errors=${base_errors}
		    if [ "${deal}" == "plain" ]
		    then
			#Example PDE/StatPDE/Example10 fails due to missing trilinos
			expected_errors=$((${expected_errors} + 1))
			#Example PDE/StatPDE/Example15 fails due to missing trilinos
			expected_errors=$((${expected_errors} + 1))
		    else if [ "${dv}" == "8.5.0" ]
			 then
			     #Example PDE/StatPDE/Example15 fails due to old deal.II
			     expected_errors=$((${expected_errors} + 1))
			 fi		    
		    fi
		    echo "" | tee -a ${log}
		    if [ ${n_errors} == ${expected_errors} ]
		    then
			echo "Passed: ${n_errors} failed as expected; in case $cs ${deal}" "${dv}" | tee -a ${log}
		    else
			echo "Failed: ${n_errors} failed but expected ${expected_errors}; in case $cs ${deal}" "${dv}" | tee -a ${log}
		    fi
		    echo "" | tee -a ${log}
		fi
	    done #deal version
	done #deal (plain/full)
    done #compiler
fi #end of no-candi tests

if [ ${candi_test} -eq 1 ]
then
    echo "Running test of current dealii with candi" | tee -a ${log}

    ##Testing Candy installation.
    #compiler loop.
    #clang doesn't work with mpi, so no real loop
    for cs in gcc 
    do 
	cd ${dir}
	
	if [ -d ${cs} ]
	then
	    echo "Subdirectory ${cs} present" | tee -a ${log}
	    directory=${dir}/${cs}
	else
	    echo "Subdirectory ${cs} not present, creating " | tee -a ${log}
	    mkdir ${cs}
	    if [ ! -d ${cs} ]
	    then 
		echo "Failed to create ${cs}" | tee -a ${log}
		exit 1
	    fi
	    directory=${dir}/${cs}
	fi
	cd ${directory}
	
	
	if [ -d candi ]
	then
	    echo "candi found, updating..." | tee -a ${log}
	    cd candi
	    git pull
	else
	    echo "Installing candi..." | tee -a ${log}
	    git clone https://github.com/dealii/candi
	    if [ ! -d candi ]
	    then 
		echo "Failed to install candi" | tee -a ${log}
		exit 1
	    fi
	    cd candi
	fi
	
	#Installation of plain deal ii
	echo "" | tee -a ${log}
	echo "Installing plain dealii " | tee -a ${log}
	echo "In directory `pwd` " | tee -a ${log}
	SET_ENV_VAR "mpi-"${cs} 
	echo "Compilers: ${CXX} ${CC} ${FC} ${FF}" | tee -a ${log}
	
	if [ -d ${directory}/${plain_deal_dir} ]
	then
	    echo "Found plain deal-ii installation, skipping recompilation" | tee -a ${log}
	    echo "To avoid this, delete the directory ${directory}/${plain_deal_dir}" | tee -a ${log}
	else
	    ./candi.sh --packages="dealii" --prefix=${directory}/${plain_deal_dir} -j ${n_procs}
	fi
	
	if [ ! -d ${directory}/${plain_deal_dir} ]
	then
	    echo "Plain deal-ii installation failed !" | tee -a ${log}
	    exit 1
	fi 
	
	cd ${directory}/${plain_deal_dir}
	
	dealii_candy_version=`ls -d deal.II-v*` 
	
	if (( $? == 0 ))
	then
	    if [ -d ${directory}/${plain_deal_dir}/${dealii_candy_version} ]
	    then
		echo "Verified deal-II installation in ${full_deal_dir-version}" | tee -a ${log}
	    else 
		echo "No directory for plain deal-ii installation found !" | tee -a ${log}
	    exit 1
	    fi
	else
	    echo "No directory for plain deal-ii installation found !" | tee -a ${log}
	    exit 1
	fi
	
	cd ${directory}/candi
	
	#Installation with MPI and everything
	echo "" | tee -a ${log}
	echo "Installing full dealii " | tee -a ${log}
	
	echo "In directory `pwd` " | tee -a ${log}
	SET_ENV_VAR "mpi-"${cs} 
	echo "Compilers: ${CXX} ${CC} ${FC} ${FF}" | tee -a ${log}
	
	if [ -d ${directory}/${full_deal_dir} ]
	then
	    echo "Found full deal-ii installation, skipping recompilation" | tee -a ${log}
	    echo "To avoid this, delete the directory ${directory}/${full_deal_dir}" | tee -a ${log}
	else
	    echo "Installing full deal-ii." | tee -a ${log}
	    ./candi.sh --prefix=${directory}/${full_deal_dir} -j ${n_procs}
	fi
	
	if [ ! -d ${directory}/${full_deal_dir} ]
	then
	    echo "Full deal-ii installation failed !" | tee -a ${log}
	    exit 1
	fi 
	
	cd ${directory}/${full_deal_dir}
	
	if [ -d ${directory}/${full_deal_dir}/${dealii_candy_version} ]
	then
	    echo "Verified full deal-II installation in ${dealii_candy_version}" | tee -a ${log}
	else 
	    echo "No directory for full deal-ii installation found !" | tee -a ${log}
	    exit 1
	fi
	
	#Installation of current deal.ii finished
	
	#Running tests
	echo "***************************************************************************" | tee -a ${log}
	echo "***************************************************************************" | tee -a ${log}
	echo "*               Running Tests for deal.ii with candi                      *" | tee -a ${log}
	echo "***************************************************************************" | tee -a ${log}
	echo "***************************************************************************" | tee -a ${log}
	for deal in plain full
	do 
	    echo "" | tee -a ${log}
	    echo "***************************************************************************" | tee -a ${log}
	    echo "Compile tests with ${deal} dealii" | tee -a ${log}
	    
	    if [ "${deal}" == "plain" ]
	    then 
		SET_ENV_VAR ${cs} 
		echo "Compilers: ${CXX} ${CC} ${FC} ${FF}" | tee -a ${log}
		export DEAL_II_DIR=${directory}/${plain_deal_dir}/${dealii_candy_version}
	    else
		if [ "${deal}" == "full" ]
		then
		    SET_ENV_VAR "mpi-"${cs} 
		    echo "Compilers: ${CXX} ${CC} ${FC} ${FF}" | tee -a ${log}
		    export DEAL_II_DIR=${directory}/${full_deal_dir}/${dealii_candy_version}
    		else
		    echo "Unknown deal version: ${deal}" | tee -a ${log}
		    exit 1
		fi
	    fi
	    
	    cd ${DOPE}
	    cd DOpEsrc
	    echo "In `pwd`, cleaning ..." | tee -a ${log}
	    make distclean
	    echo "building..." | tee -a ${log}
	    (make c-all -j${n_procs} 2>&1) | tee ${dir}/${deal}_candi_${dealii_candy_version}_${cs}.make
	    if [ ${PIPESTATUS[0]} -eq 0 ]
	    then
		echo "Build passed for "$cs" "${deal} | tee -a ${log}
	    else
		echo "Build failed for "$cs" "${deal} | tee -a ${log}
		break
	    fi
	    cd ${DOPE}
	    cd Examples 
	    echo "In `pwd`, building ..." | tee -a ${log}
	    (make c-all -j${n_procs} 2>&1) | tee ${dir}/${deal}_candi_${dealii_candy_version}_${cs}.make-examples
	    if [ ${PIPESTATUS[0]} -eq 0 ]
	    then
		echo "Example Build passed for "$cs" "${deal} | tee -a ${log}
	    else
		echo "Example Build failed for "$cs" "${deal} | tee -a ${log}
		break
	    fi
	    echo "In `pwd`, testing ..." | tee -a ${log}
	    if [ ! -f ./testtime.sh ]
	    then 
		echo "Missing Testscript." | tee -a ${log}
		exit 1
	    fi
	    (./testtime.sh -j${n_procs} 2>&1) | tee -a ${dir}/${deal}_candi_${dealii_candy_version}_${cs}.tests
	    n_errors=${PIPESTATUS[0]}
	    #Maybe some fail if no SNOPT or IPOPT are given
	    expected_errors=${base_errors}
	    if [ "${deal}" == "plain" ]
	    then
		#Example PDE/StatPDE/Example10 fails due to missing trilinos
		expected_errors=$((${expected_errors} + 1))
		#Example PDE/StatPDE/Example15 fails due to missing trilinos
		expected_errors=$((${expected_errors} + 1))
	    fi
	    echo "" | tee -a ${log}
	    if [ ${n_errors} == ${expected_errors} ]
	    then
		echo "Passed: ${n_errors} failed as expected; in case $cs ${deal}" | tee -a ${log}
	    else
		echo "Failed: ${n_errors} failed but expected ${expected_errors}; in case $cs ${deal}" | tee -a ${log}
	    fi
	    echo "" | tee -a ${log}
	done #deal (plain/full)
    done #compiler
fi #candi-tests
