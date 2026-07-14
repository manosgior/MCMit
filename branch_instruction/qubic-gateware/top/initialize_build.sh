#!/bin/bash

mkdir $1
cp src/dsp_config.yaml ${1}
cp -r src/cocotb ${1}
cd $1
ln -s ../configure_build.sh configure_build.sh
