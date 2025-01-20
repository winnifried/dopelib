#!/bin/bash
#script installs deal.ii with all needed dependencies
if (( $# == 0 ))
then
    echo "Missing options, run 'compile-tests.sh -h' for help."
    exit 1
fi

dir=`pwd`
n_procs=1

while getopts 'hd:j:apv:m:' flag; do
    case "${flag}" in
	h) helpflag=1 ;;
	d) dir=`realpath ${OPTARG}` ;;
	j) n_procs="${OPTARG}" ;;
	a) fullinstall=1 ;;
	p) plaininstall=1 ;;
	v) dealii_version="${OPTARG}" ;;
	m) mpi_package="${OPTARG}" ;;
	*) echo "Unknown option ${flag}."
	   echo "Run 'compile-tests.sh -h' for help."
	   exit 1 ;;
    esac
done

if (( ${helpflag} ))
then
    echo "Usage: $0 [-h] -d [pathname] [-j [n_procs]]"
    echo "-h : Show this help"
    echo "-d : Path where the installations should go to"
    echo "-a : If dealii should be installed with third-party libraries 
               Trilinos, Petsc, Slepc, P4est, MPI"
    echo "-p : if dealii should be installed without third-party libraries"
    echo "-v : Specify Version of deal.ii to be installed, e.g., [git|9.3.0|...]"
    echo "-m : Specify MPI package [openmpi,mpich,system]"
    echo "-j : Number of Processes to be used"
    exit 0
fi

if (( ${fullinstall} )) && (( ${plaininstall} ))
then
    echo "You can not specify both full and plain install"
    exit 1
fi
if ! {(( ${fullinstall} )) || (( ${plaininstall} )); }
then
    echo "You must specify full or plain install"
    exit 1
fi

if (( ${fullinstall} ))
then
    dealii_install_type=full
    if [ "${mpi_package}" == "mpich" ]
    then
	echo "Selecting mpich"
	if [ `uname` == "Darwin" ]
	then
	    export CXX=mpicxx-mpich-mp
	    export CC=mpicc-mpich-mp
	    export FC=mpif90-mpich-mp
	    export FF=mpif77-mpich-mp
	    export F77=mpif77-mpich-mp
	    export MPIEXEC=mpiexec-mpich-mp
	else
	    export CXX=mpicxx.mpich
	    export CC=mpicc.mpich
	    export FC=mpif90.mpich
	    export FF=mpif77.mpich
	    export MPIEXEC=mpiexec.mpich
	fi
    else if [ "${mpi_package}" == "openmpi" ]
	 then
	     echo "Selecting openmpi"
	     export CXX=mpicxx.openmpi
	     export CC=mpicc.openmpi
	     export FC=mpif90.openmpi
	     export FF=mpif77.openmpi
	     export MPIEXEC=mpiexec.openmpi
	 else if [ "${mpi_package}" == "system" ]
	      then 
		  echo "Selecting system default mpi"
		  export CXX=mpicxx
		  export CC=mpicc
		  export FC=mpif90
		  export FF=mpif77
		  export MPIEXEC=mpiexec
	      else
		  echo "Unknown mpi compiler ${mpi_package}"
	      fi	     
	 fi
    fi
else
    dealii_install_type=plain
    export CXX=g++
    export CC=gcc
    export FC=gfortran
    export FF=gfortran
fi

if [ "${dealii_version}" == "git" ] || [ "${dealii_version}" == "9.6.2" ] || [ "${dealii_version}" == "9.6.1" ] || [ "${dealii_version}" == "9.6.0" ] || [ "${dealii_version}" == "9.5.2" ]|| [ "${dealii_version}" == "9.5.1" ] || [ "${dealii_version}" == "9.5.0" ] 
then
    trilinos_version=14-4-0
    p4est_version=2.0
    scalapack_version=2.2.2
    
    petsc_version=3.20.0
    petsc_debug=1  #build debug version of PETSc (1 recommended)
    slepc_version=3.20.0 #must be compatible, see http://slepc.upv.es/download/changes.htm
    
    PYTHON=python3 #Python Version needed by petsc	
else 
    if [ "${dealii_version}" == "9.4.2" ] || ["${dealii_version}" == "9.4.1" ] || [ "${dealii_version}" == "9.4.0" ] || [ "${dealii_version}" == "9.3.3" ] || [ "${dealii_version}" == "9.3.2" ] || [ "${dealii_version}" == "9.3.1" ] || [ "${dealii_version}" == "9.3.0" ] || [ "${dealii_version}" == "9.2.0" ] || [ "${dealii_version}" == "9.1.1" ] || [ "${dealii_version}" == "9.0.1" ] 
    then
	#dealii_version=9.1.1
	#dealii_version=9.2.0
	#dealii_version=9.3.0
	#dealii_version=git
	#trilinos_version=12-18-1
	trilinos_version=12-12-1
	p4est_version=2.0
	scalapack_version=2.2.2
	
	#petsc_version=3.9.4
	#petsc_version=3.12.4
	#petsc_version=3.18.5
	petsc_version=3.20.0
	petsc_debug=1  #build debug version of PETSc (1 recommended)
	#slepc_version=3.9.2 #must be compatible, see http://slepc.upv.es/download/changes.htm
	#slepc_version=3.12.2 #must be compatible, see http://slepc.upv.es/download/changes.htm
	#slepc_version=3.18.2 #must be compatible, see http://slepc.upv.es/download/changes.htm
	slepc_version=3.20.0 #must be compatible, see http://slepc.upv.es/download/changes.htm
	
	#PYTHON=python2.7 #Python Version needed by petsc 3.12
	PYTHON=python3 #Python Version needed by petsc
	
	#Old Versions of dealii without petsc
	if [ "${dealii_version}" == "9.0.1" ] || [ "${dealii_version}" == "8.5.0" ] || [ "${dealii_version}" == "8.4.1" ] || [ "${dealii_version}" == "8.3.0" ]
	then
	    no_petsc=1
	fi
	
    else
	echo "Unknown deal.ii version "${dealii_version}" must be [git|9.6.2|9.6.1|9.6.0|9.5.2|9.5.1|9.5.0|9.4.2|9.4.1|9.4.0|9.3.3|9.3.2|9.3.1|9.3.0|9.2.0|9.1.1|9.0.1]"
	exit 1;
    fi
fi

echo "Starting installation in "${dir}

#Check base-dir existence
if [[ ! -d $dir ]]
then
    mkdir $dir
    if [[ ! -d $dir ]]
    then
	echo "Failed to create install directory "$dir
	exit 1
    else
	echo $dir" created."
    fi
fi

if (( ${fullinstall} ))
then
    log=${dir}/dealii-${dealii_version}-full-install-log_`date +%Y%m%d`.txt
else
    log=${dir}/dealii-${dealii_version}-plain-install-log_`date +%Y%m%d`.txt
fi
   
if [ -f ${log} ]
then
    echo "Found previous log file, deleting..."
    rm ${log}
fi
touch ${log}

cd $dir
if [[ ! -d sources ]]
then
    mkdir sources
    if [[ ! -d sources ]]
    then
	echo "Failed to create 'sources' directory "
	exit 1
    else
	echo "'sources' created."
    fi
fi

if [[ ! -d builds ]]
then
    mkdir builds
    if [[ ! -d builds ]]
    then
	echo "Failed to create 'builds' directory "
	exit 1
    else
	echo "'builds' created."
    fi
fi

if (( ${fullinstall} ))
then
    #Install p4est
    cd $dir
    #Only if needed!
    if  [[ ! -d p4est-${p4est_version}-install ]]
    then
	cd sources
	if [[ ! -f p4est-${p4est_version}.tar.gz ]]
	then
	    wget https://github.com/p4est/p4est.github.io/raw/master/release/p4est-${p4est_version}.tar.gz
	    if [[ ! -f p4est-${p4est_version}.tar.gz ]]
	    then
		echo "Failed to download p4est ${p4est_version}"
		exit 1
	    fi
	fi
	if [[ ! -f p4est-setup.sh ]]
	then
	    wget http://www.dealii.org/current/external-libs/p4est-setup.sh
	    if [[ ! -f p4est-setup.sh ]]
	    then
		echo "Failed to download p4est-setup.sh"
		exit 1
	    fi
	fi
	chmod u+x p4est-setup.sh
	
	cd $dir
	cd builds
	../sources/p4est-setup.sh ../sources/p4est-${p4est_version}.tar.gz ${dir}/p4est-${p4est_version}-install
	if [ ! $? -eq 0 ]
	then
	    echo "p4est build failed!" | tee -a ${log}
	    cd ${dir}
	    rm -r p4est-${p4est_version}-install
	    exit 1
	fi
	echo "P4est ${p4est_version} build succeeded!" | tee -a ${log}
    else
	echo "P4est ${p4est_version} already installed" | tee -a ${log}
    fi
    
    #Install trilinos
    cd $dir
    #Only if needed!
    if  [[ ! -d trilinos-${trilinos_version}-install ]]
    then
	cd sources
	if [[ ! -d Trilinos ]]
	then
	    git clone https://github.com/trilinos/Trilinos
	fi
	if [[ ! -d Trilinos ]]
	then
	    echo "Failed to clone Trilinos git"
	    exit 1
	fi
	cd Trilinos
	git checkout trilinos-release-${trilinos_version}
	cd $dir
	cd builds
	if [[ ! -d trilinos-${trilinos_version}-build ]]
	then
	    mkdir trilinos-${trilinos_version}-build
	fi
	if [[ ! -d trilinos-${trilinos_version}-build ]]
	then
	    echo "Failed to create trilinos build dir"
	    exit 1
	fi
	if [[ ! -d ${dir}/trilinos-${trilinos_version}-install ]]
	then
	    mkdir ${dir}/trilinos-${trilinos_version}-install
	fi
	if [[ ! -d ${dir}/trilinos-${trilinos_version}-install ]]
	then
	    echo "Failed to create trilinos install dir"
	    exit 1
	fi
	cd trilinos-${trilinos_version}-build
	cmake -DTrilinos_ENABLE_Amesos=ON -DTrilinos_ENABLE_Epetra=ON -DTrilinos_ENABLE_EpetraExt=ON -DTrilinos_ENABLE_Ifpack=ON -DTrilinos_ENABLE_AztecOO=ON -DTrilinos_ENABLE_Sacado=ON -DTrilinos_ENABLE_Teuchos=ON  -DTrilinos_ENABLE_MueLu=ON -DTrilinos_ENABLE_ML=ON -DTrilinos_ENABLE_ROL=ON -DTrilinos_ENABLE_Tpetra=ON  -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON -DTrilinos_ENABLE_Zoltan=ON -DTrilinos_VERBOSE_CONFIGURE=OFF -DTPL_ENABLE_MPI=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=RELEASE -DMPI_C_COMPILER:FILEPATH=$CC -DMPI_CXX_COMPILER:FILEPATH=$CXX -DMPI_Fortran_COMPILER:FILEPATH=$FC -DCMAKE_INSTALL_PREFIX:PATH=${dir}/trilinos-${trilinos_version}-install ${dir}/sources/Trilinos
	if [ ! $? -eq 0 ]
	then
	    echo "Trilinos cmake config failed!" | tee -a ${log}
	    cd ${dir}
	    rm -r builds/trilinos-${trilinos_version}-build
	    rm -r trilinos-${trilinos_version}-install
	    exit 1
	fi
	make install -j ${n_procs}
	if [ ! $? -eq 0 ]
	then
	    echo "Trilinos install failed!" | tee -a ${log}
	    cd ${dir}
	    rm -r builds/trilinos-${trilinos_version}-build
	    rm -r trilinos-${trilinos_version}-install
	    exit 1
	fi
	echo "Trilinos ${trilinos_version} build succeeded!" | tee -a ${log}    
    else
	echo "Trilinos ${trilinos_version} already installed" | tee -a ${log}
    fi
#Install scalapack
    cd ${dir}
    if  [[ ! -d scalapack-${scalapack_version}-install ]]
    then
	cd sources
	if [[ ! -f scalapack-${scalapack_version}.tgz  ]]
	then
	    wget http://www.netlib.org/scalapack/scalapack-${scalapack_version}.tgz
	fi
	if [[ ! -f scalapack-${scalapack_version}.tgz  ]]
	then
	    echo "Failed to unpack SCALAPACK ${scalapack_version}"
	exit 1
	fi
	tar xvfz scalapack-${scalapack_version}.tgz
	if [[ ! -d scalapack-${scalapack_version} ]]
	then
	    echo "Failed to unpack SCALAPACK ${scalapack_version}"
	    exit 1
	fi
	cd $dir
	cd builds
	if [[ ! -d scalapack-${scalapack_version}-build ]]
	then
	    mkdir scalapack-${scalapack_version}-build
	fi
	if [[ ! -d scalapack-${scalapack_version}-build ]]
	then
	    echo "Failed to create SCALAPACK build dir"
	    exit 1
	fi
	if [[ ! -d ${dir}/scalapack-${scalapack_version}-install ]]
	then
	    mkdir ${dir}/scalapack-${scalapack_version}-install
	fi
	if [[ ! -d ${dir}/scalapack-${scalapack_version}-install ]]
	then
	    echo "Failed to create SCALAPACK install dir"
	    exit 1
	fi
	cd scalapack-${scalapack_version}-build
	cmake -DCMAKE_INSTALL_PREFIX=${dir}/scalapack-${scalapack_version}-install -DBUILD_SHARED_LIBS=ON -DMPI_C_COMPILER=${CC} -DMPI_Fortran_COMPILER=$FC ${dir}/sources/scalapack-${scalapack_version}
	if [ ! $? -eq 0 ]
	then
	    echo "SCALAPACK cmake config failed!" | tee -a ${log}
	    cd ${dir}
	    rm -r builds/scalapack-${scalapack_version}-build
	    rm -r scalapack-${scalapack_version}-install
	    exit 1
	fi
	make install -j ${n_procs}
	if [ ! $? -eq 0 ]
	then
	    echo "SCALAPACK install failed!" | tee -a ${log}
	    cd ${dir}
	    rm -r builds/scalapack-${scalapack_version}-build
	    rm -r scalapack-${scalapack_version}-install
	    exit 1
	fi
	echo "SCALAPACK ${scalapack_version} build succeeded!" | tee -a ${log}    
    else
	echo "SCALAPACK ${scalapack_version} already installed" | tee -a ${log}
    fi

    if ! (( ${no_petsc} ))
    then
	#install petsc
	cd ${dir}
	if [[ ! -d petsc-${petsc_version}-install ]]
	then
	    cd sources
	    if [[ ! -f petsc-${petsc_version}.tar.gz ]]
	    then
		#wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-${petsc_version}.tar.gz
		wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${petsc_version}.tar.gz
	    fi
	    if [[ ! -f petsc-${petsc_version}.tar.gz ]]
	    then
		echo "Failed to download petsc ${petsc_version}"
		exit 1
	    fi
	    tar xvzf petsc-${petsc_version}.tar.gz
	    if [[ ! -d petsc-${petsc_version} ]]
	    then
		echo "Failed to unpack petsc ${petsc_version}"
		exit 1
	    fi
	    #petsc is configured in source!!!
	    
	    cd petsc-${petsc_version}
	    export PETSC_DIR=`pwd`
	    $PYTHON ./config/configure.py --with-shared=1 --with-x=0 --with-mpi=1 --download-hypre=1 --download-f2cblaslapack --with-debugging=${petsc_debug} CC=$CC CXX=$CXX FC=$FC FF=$FF --with-mpiexec=$MPIEXEC --prefix=${dir}/petsc-${petsc_version}-install 
	    if [ ! $? -eq 0 ]
	    then
		echo "PETSC config failed!" | tee -a ${log}
		cd ${dir}
		rm -r sources/petsc-${petsc_version}
		rm -r petsc-${petsc_version}-install
		exit 1
	    fi
	    make -j ${n_procs}
	    make test
	    if [ ! $? -eq 0 ]
	    then
		echo "PETSC tests failed!" | tee -a ${log}
		cd ${dir}
		rm -r sources/petsc-${petsc_version}
		rm -r petsc-${petsc_version}-install
		exit 1
	    fi
	    make install
	    if [ ! $? -eq 0 ]
	    then
		echo "PETSC install failed!" | tee -a ${log}
		cd ${dir}
		rm -r sources/petsc-${petsc_version}
		rm -r petsc-${petsc_version}-install
		exit 1
	    fi
	    export PETSC_DIR=${dir}/petsc-${petsc_version}-install/
	    echo "PETSC ${petsc_version} build successfully" | tee -a ${log}
	else
	    export PETSC_DIR=${dir}/petsc-${petsc_version}-install/
	    echo "PETSC ${petsc_version} already installed" | tee -a ${log}
	fi
	
	#install slepc
	cd ${dir}
	if [[ ! -d slepc-${slepc_version}-install ]]
	then
	    cd sources
	    if [[ ! -f slepc-${slepc_version}.tar.gz ]]
	    then
		wget slepc.upv.es/download/distrib/slepc-${slepc_version}.tar.gz
	    fi
	    if [[ ! -f slepc-${slepc_version}.tar.gz ]]
	    then
		echo "Failed to download slepc ${slepc_version}"
		exit 1
	    fi
	    tar xvzf slepc-${slepc_version}.tar.gz
	    if [[ ! -d slepc-${slepc_version} ]]
	    then
		echo "Failed to unpack slepc ${slepc_version}"
		exit 1
	    fi
	    #slepc is configured in source!!!
	    
	    cd slepc-${slepc_version}
	    export SLEPC_DIR=`pwd`
	    $PYTHON ./configure --prefix=${dir}/slepc-${slepc_version}-install
	    if [ ! $? -eq 0 ]
	    then
		echo "SLEPC config failed!" | tee -a ${log}
		cd ${dir}
		rm -r sources/slepc-${slepc_version}
		rm -r slepc-${slepc_version}-install
		exit 1
	    fi
	    make -j ${n_procs}
	    make install
	    if [ ! $? -eq 0 ]
	    then
		echo "SLEPC install failed!" | tee -a ${log}
		cd ${dir}
		rm -r sources/slepc-${slepc_version}
		rm -r slepc-${slepc_version}-install
		exit 1
	    fi
	    export SLEPC_DIR=${dir}/slepc-${slepc_version}-install/
	    echo "SLEPC ${slepc_version} build successfully" | tee -a ${log}
	else
	    export SLEPC_DIR=${dir}/slepc-${slepc_version}-install/
	    echo "SLEPC ${slepc_version} already installed" | tee -a ${log}
	fi
    fi # endof petsc/slepc install
fi #endof preinstall for fullinstallation

#Install deal.II
cd ${dir}
if [ "${dealii_version}" == "git" ]
then
    if  [[ -d dealii-${dealii_version}-${dealii_install_type}-install ]]
    then
	rm -r dealii-${dealii_version}-${dealii_install_type}-install
    fi
fi

if  [[ ! -d dealii-${dealii_version}-${dealii_install_type}-install ]]
then
    cd sources
    if [ "${dealii_version}" == "git" ]
    then
	if [ ! -d dealii-${dealii_version} ]
	then
	    git clone https://github.com/dealii/dealii.git dealii-${dealii_version}
	else
	    cd dealii-${dealii_version}
	    git pull
	    cd ..
	fi
    else
	if [[ ! -f dealii-${dealii_version}.tar.gz  ]]
	then
	    wget https://github.com/dealii/dealii/releases/download/v${dealii_version}/dealii-${dealii_version}.tar.gz
	fi
	if [[ ! -f dealii-${dealii_version}.tar.gz  ]]
	then
	    echo "Failed to download dealii ${dealii_version}"
	    exit 1
	fi
	tar xvfz dealii-${dealii_version}.tar.gz
	if [[ ! -d dealii-${dealii_version} ]]
	then
	    echo "Failed to unpack dealii ${dealii_version}"
	    exit 1
	fi
    fi
    
    cd $dir
    cd builds
    if [[ ! -d dealii-${dealii_version}-${dealii_install_type}-build ]]
    then
	mkdir dealii-${dealii_version}-${dealii_install_type}-build
    fi
    if [[ ! -d dealii-${dealii_version}-${dealii_install_type}-build ]]
    then
	echo "Failed to create dealii build dir"
	exit 1
    fi
    if [[ ! -d ${dir}/dealii-${dealii_version}-${dealii_install_type}-install ]]
    then
	mkdir ${dir}/dealii-${dealii_version}-${dealii_install_type}-install
    fi
    if [[ ! -d ${dir}/dealii-${dealii_version}-${dealii_install_type}-install ]]
    then
	echo "Failed to create dealii install dir"
	exit 1
    fi
    cd dealii-${dealii_version}-${dealii_install_type}-build
    if (( ${fullinstall} ))
    then
	if (( ${no_petsc} ))
	then
	    cmake -DCMAKE_INSTALL_PREFIX=${dir}/dealii-${dealii_version}-${dealii_install_type}-install/ -DDEAL_II_WITH_UMFPACK=true -DDEAL_II_FORCE_BUNDLED_UMFPACK=true -DDEAL_II_WITH_TRILINOS=true -DTRILINOS_DIR=${dir}/trilinos-${trilinos_version}-install/ -DDEAL_II_WITH_MPI=true -DDEAL_II_WITH_P4EST=true -DP4EST_DIR=${dir}/p4est-${p4est_version}-install/ -DSCALAPACK_DIR=${dir}/scalapack-${scalapack_version}-install -DDEAL_II_WITH_PETSC=OFF -DDEAL_II_WITH_SLEPC=OFF -DMPI_C_COMPILER:FILEPATH=$CC -DMPI_CXX_COMPILER:FILEPATH=$CXX -DMPI_Fortran_COMPILER:FILEPATH=$FC  $dir/sources/dealii-${dealii_version}/
	else
	    cmake -DCMAKE_INSTALL_PREFIX=${dir}/dealii-${dealii_version}-${dealii_install_type}-install/ -DDEAL_II_WITH_UMFPACK=true -DDEAL_II_FORCE_BUNDLED_UMFPACK=true -DDEAL_II_WITH_TRILINOS=true -DTRILINOS_DIR=${dir}/trilinos-${trilinos_version}-install/ -DDEAL_II_WITH_MPI=true -DDEAL_II_WITH_P4EST=true -DP4EST_DIR=${dir}/p4est-${p4est_version}-install/ -DSCALAPACK_DIR=${dir}/scalapack-${scalapack_version}-install -DDEAL_II_WITH_PETSC=ON -DPETSC_DIR=${dir}/petsc-${petsc_version}-install/ -DPETSC_ARCH=$PETSC_ARCH -DSLEPC_DIR=$dir/slepc-${slepc_version}-install -DDEAL_II_WITH_SLEPC=ON -DMPI_C_COMPILER:FILEPATH=$CC -DMPI_CXX_COMPILER:FILEPATH=$CXX -DMPI_Fortran_COMPILER:FILEPATH=$FC  $dir/sources/dealii-${dealii_version}/
	fi
    else
	cmake -DCMAKE_INSTALL_PREFIX=${dir}/dealii-${dealii_version}-${dealii_install_type}-install/ -DDEAL_II_WITH_UMFPACK=true -DDEAL_II_FORCE_BUNDLED_UMFPACK=true -DDEAL_II_WITH_TRILINOS=FALSE -DDEAL_II_WITH_MPI=FALSE -DDEAL_II_WITH_P4EST=FALSE -DDEAL_II_WITH_PETSC=OFF -DDEAL_II_WITH_SLEPC=OFF  $dir/sources/dealii-${dealii_version}/
    fi
    if [ ! $? -eq 0 ]
    then
	echo "dealii cmake config failed!" | tee -a ${log}
	cd ${dir}
	rm -r builds/dealii-${dealii_version}-${dealii_install_type}-build
	rm -r dealii-${dealii_version}-${dealii_install_type}-install
	exit 1
    fi
    make install -j ${n_procs}
    if [ ! $? -eq 0 ]
    then
	echo "dealii install failed!" | tee -a ${log}
	cd ${dir}
	rm -r builds/dealii-${dealii_version}-${dealii_install_type}-build
	rm -r dealii-${dealii_version}-${dealii_install_type}-install
	exit 1
    fi
    echo "dealii ${dealii_version} build succeeded!" | tee -a ${log}    
else
    echo "dealii ${dealii_version} already installed" | tee -a ${log}
fi

echo "Installations complete. You may wish to export the following paths:" | tee -a ${log}
echo " export DEAL_II_DIR="${dir}/dealii-${dealii_version}-${dealii_install_type}-install/ | tee -a ${log}
echo " export SCALAPACK_DIR="${dir}/scalapack-${scalapack_version}-install/lib/ | tee -a ${log}
echo " export PETSC_DIR="${dir}/petsc-${petsc_version}-install/ | tee -a ${log}
echo " export SLEPC_DIR="${dir}/slepc-${slepc_version}-install/ | tee -a ${log}
echo " export CXX="$CXX | tee -a ${log}
echo " export  CC="$CC | tee -a ${log}
echo " export  FC="$FC | tee -a ${log}
echo " export  FF="$FF | tee -a ${log}
echo " export  MPI_CXX_COMPILER="$CXX | tee -a ${log}
echo " export  MPI_C_COMPILER="$CC | tee -a ${log}
echo " export  MPI_Fortran_COMPILER="$FC | tee -a ${log}
exit 0
