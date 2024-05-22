#!/bin/bash

ORIGINALDIR=/content/app
# Use predefined DATADIR if it is defined
[[ x"${DATADIR}" == "x" ]] && DATADIR=/content/data

# Make persistent dir from original dir
function mklink () {
	mkdir -p $DATADIR/$1
	ln -s $DATADIR/$1 $ORIGINALDIR
}

# Copy old files from import dir
function import () {
	(test -d /import/$1 && cd /import/$1 && cp -Rpn . $DATADIR/$1/)
}

cd $ORIGINALDIR

# models
mklink models
# Copy original files
(cd $ORIGINALDIR/models.org && cp -Rpn . $ORIGINALDIR/models/)
# Import old files
import models

# outputs
mklink outputs
# Import old files
import outputs

# Start application
python launch.py $*
