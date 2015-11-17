#!/bin/bash 
#installation of the third party lib ipopt 
#to a given location.
####################################################################
#
# Copyright (C) 2012 by the DOpElib authors
#
# This file is part of DOpElib
#
# DOpElib is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later
# version.
#
# DOpElib is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the GNU General Public License for more
# details.
#
# Please refer to the file LICENSE.TXT included in this distribution
# for further information on this license.
#
#
####################################################################
#installation of ipopt
####################################################################
#ARCHIVE_BASE=Ipopt-3.10.4
#ARCHIVE_BASE=Ipopt-3.11.8
ARCHIVE_BASE=Ipopt-3.12.4
ARCHIVE=${ARCHIVE_BASE}.tgz


echo "Current directory: "`pwd`
echo "Press enter to install here or provide path?"
read line
 IPATH="ipopt"
if [[ $line != "" ]]
then
    cd $line
fi
echo "Installation of ipopt into directory: "`pwd`"/$IPATH"
echo "[y/n]"
read line
if [[ $line != "y" ]]
then
    #only if all is fine we continue
    echo "Stopping script!"
    exit 1
fi
INST_DIR=`pwd`/$IPATH

if [[ -f ${ARCHIVE} ]]
then 
    echo "Reusing existing archive"
else
    wget http://www.coin-or.org/download/source/Ipopt/${ARCHIVE}
fi

if [[ -f $ARCHIVE ]]
then
    echo "Download completed."
else
    echo "Download failed."
    exit 1
fi


if [[ -d $IPATH ]] 
then 
    echo "There seems to be an existing ipopt installation"
    echo "overwrite [y/n]"
    read line
    if [[ $line != "y" ]]
    then
	#reuse?
	echo "Do you want to reuse the existing installation?"
	echo "[y/n]"
	read line
	if [[ $line != "y" ]]
	then
           #only if all is fine we continue
	    echo "Stopping script!"
	    exit 1
	else
	    cd $IPATH
	fi
    else
	echo "deleting directory"
	rm -r $IPATH
	{
	    tar xvfz $ARCHIVE 
	} <&- #avoid reading stdin
	mv $ARCHIVE_BASE $IPATH
	if [[ -d $IPATH ]]
	then
	    cd $IPATH
	else
	    echo "Failed to extract archive"
	fi
    fi
else
    {
	tar xvfz $ARCHIVE
    } <&- #avoid reading stdin
    mv $ARCHIVE_BASE $IPATH
	
    if [[ -d $IPATH ]]
    then
	cd $IPATH
    else
	echo "Failed to extract archive"
    fi
fi

#Obtain the required solvers...
#ASL
cd ThirdParty/ASL
if [[ -d solvers ]]
then
    echo "The ASL Solver appears to be available already."
    echo "reuse existing code? [y/n]"
    read line
    if [[ $line != "y" ]]
    then
	echo "Ipopt needs additional solvers:"
	echo "Download the ASL Solver [y/n]?"
	read line
	if [[ $line != "y" ]]
	then
	    echo "Stopping script!"
	    exit 1
	fi
	./get.ASL
    fi
else
    echo "Ipopt needs additional solvers:"
    echo "Download the ASL Solver [y/n]?"
    read line
    if [[ $line != "y" ]]
    then
	echo "Stopping script!"
	exit 1
    fi
    ./get.ASL
fi
#back to the ${INST_DIR}
cd ../..
#done ASL installation
#Metis
cd ThirdParty/Metis
if [[ -d metis-4.0 ]]
then
    echo "The Metis library appears to be available already."
    echo "reuse existing code? [y/n]"
    read line
    if [[ $line != "y" ]]
    then
	echo "Ipopt needs additional solvers:"
	echo "Download the Metis library [y/n]?"
	read line
	if [[ $line != "y" ]]
	then
	    echo "Stopping script!"
	    exit 1
	fi
	./get.Metis
    fi
else
    echo "Ipopt needs additional solvers:"
    echo "Download the Metis library [y/n]?"
    read line
    if [[ $line != "y" ]]
    then
	echo "Stopping script!"
	exit 1
    fi
    ./get.Metis
fi
#back to the ${INST_DIR}
cd ../..
#done Metis installation
#Mumps
cd ThirdParty/Mumps
if [[ -d MUMPS ]]
then
    echo "The MUMPS Solver appears to be available already."
    echo "reuse existing code? [y/n]"
    read line
    if [[ $line != "y" ]]
    then
	echo "Ipopt needs additional solvers:"
	echo "Download the MUMPS Solver [y/n]?"
	read line
	if [[ $line != "y" ]]
	then
	    echo "Stopping script!"
	    exit 1
	fi
	./get.Mumps
    fi
else
    echo "Ipopt needs additional solvers:"
    echo "Download the MUMPS Solver [y/n]?"
    read line
    if [[ $line != "y" ]]
    then
	echo "Stopping script!"
	exit 1
    fi
    ./get.Mumps
fi
#back to the ${INST_DIR}
cd ../..
#done Mumps installation
#Lapack
cd ThirdParty/Lapack
if [[ -d LAPACK ]]
then
    echo "The Lapack library appears to be available already."
    echo "reuse existing code? [y/n]"
    read line
    if [[ $line != "y" ]]
    then
	echo "Ipopt needs additional solvers:"
	echo "Download the Lapack library [y/n]?"
	read line
	if [[ $line != "y" ]]
	then
	    echo "Stopping script!"
	    exit 1
	fi
	./get.Lapack
    fi
else
    echo "Ipopt needs additional solvers:"
    echo "Download the Lapack library [y/n]?"
    read line
    if [[ $line != "y" ]]
    then
	echo "Stopping script!"
	exit 1
    fi
    ./get.Lapack
fi
#back to the ${INST_DIR}
cd ../..
#done Lapack installation

#HSL
echo "Please install/or download appropriate linear solvers"
#echo "This script has been tested with HSL MA 27 and HSL MA 77"
echo "See the file "${INST_DIR}"/ThirdParty/HSL/INSTALL.HSL"
echo "on how to obtain the required sources. "
#echo "Copy the required files to "${INST_DIR}"/ThirdParty/HSL/:"
#echo "For MA27 copy the file ma27d.f to ma27ad.f"
#echo "For MA77 see the INSTALL.HSL file"
echo ""
echo "Do you wish to continue? [y/n]"
read line
if [[ $line != "y" ]]
then
    #only if all is fine we continue
    echo "Stopping script!"
    exit 1
fi
#Done with HSL

#done with solvers
echo ""
echo "Starting the configuration!"
echo ""
./configure -enable-static -with-asl -with-mumps -with-hsl --enable-dependency-linking

read -t 1 -n 10000 discard #remove unwanted input
echo "Configuration complete:"
echo "proceede [y/n]"
read line
if [[ $line != "y" ]]
then
    #only if all is fine we continue
    echo "Stopping script!"
    exit 1
fi

#build the library
make 

read -t 1 -n 10000 discard #remove unwanted input
echo "Build complete:"
echo "proceede with tests [y/n]"
read line
if [[ $line != "y" ]]
then
    #only if all is fine we continue
    echo "No Checks are done!"
else
    make test
fi

read -t 1 -n 10000 discard #remove unwanted 
echo "proceede with installation [y/n]"
read line
if [[ $line != "y" ]]
then
    #only if all is fine we continue
    echo "Stopping script!"
    exit 0
fi
make install

if [[ -d ${INST_DIR}/lib64 ]]
then
    echo ""
    echo "**************************************************************"
    echo "                 Installation complete!"
    echo "Add "${INST_DIR}/lib64
    echo "    to your \$LD_LIBRARY_PATH variable"
    echo "**************************************************************"
else
    #try if it is called lib
    if [[ -d ${INST_DIR}/lib ]]
    then
	mv ${INST_DIR}/lib ${INST_DIR}/lib64
	echo ""
	echo "**************************************************************"
	echo "                 Installation complete!"
	echo "Add "${INST_DIR}/lib64
	echo "    to your \$LD_LIBRARY_PATH variable"
	echo "**************************************************************"
    else
	echo ""
	echo "**************************************************************"
	echo "                 Installation failed!"
	echo "No Library Directory in "${INST_DIR}/" detected!"
	echo "**************************************************************"
    fi
fi

#endof ipopt installation
exit 0
