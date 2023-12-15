#!/bin/bash

ORIGINALDIR=/content/app
# Use predefined DATADIR if it is defined
[[ x"${DATADIR}" == "x" ]] && DATADIR=/content/data

# make persistent dir from original dir
function mklink () {
	mkdir -p $DATADIR/$1
	ln -s $DATADIR/$1 $ORIGINALDIR
}

cd $ORIGINALDIR

# models
mklink models
# Copy original files
(cd $ORIGINALDIR/models.org && cp -Rpn . $ORIGINALDIR/models/)

# outputs
mklink outputs

# Start application
python entry_with_update.py $*
