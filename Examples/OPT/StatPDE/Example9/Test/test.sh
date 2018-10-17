#!/bin/bash
if [ $# -ne 1 ]
    then
    echo "Usage: "$0" [Test|Store]"
    exit 1
fi

PROGRAM=../DOpE-OPT-StatPDE-Example9

../../../../test-single.sh $1 $PROGRAM
