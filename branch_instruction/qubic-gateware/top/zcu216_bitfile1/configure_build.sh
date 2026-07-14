#!/bin/bash
# usage: ./configure_build.sh <config_file>
# this should be run from the top directory
# creates a build directory in top given by <build_name>,
# then configures dsp cores according to the provided
# <config_file> (should be a .yaml)

DATETIME=$(date +'%Y%m%d%H%M%S')
COMMITNUM=$(git log --pretty=oneline --abbrev-commit --abbrev=8|head -1 | awk '{print $1}')

if [ "$2" == 'sim' ]; then
    BUILDDIR=sim
else
    BUILDDIR=build_${COMMITNUM}_${DATETIME}
fi
mkdir "$BUILDDIR"
cp -r ../src/zcu216/* "$BUILDDIR"
cp $1 $BUILDDIR
if [ "$2" == 'sim' ]; then
    python ../src/dsp_build.py $BUILDDIR $1 --sim
else
    python ../src/dsp_build.py $BUILDDIR $1
fi

