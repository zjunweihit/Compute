#!/bin/bash

opt=$1

if [ "${opt}" == "all" ]; then
	ln -sf shdma-all.log shdma.log
	ln -sf cpdma-all.log cpdma.log
	ln -sf sdma-all.log sdma.log
	echo "change to test all mode"
elif [ "${opt}" == "big" ]; then
	ln -sf shdma-big.log shdma.log
	ln -sf cpdma-big.log cpdma.log
	ln -sf sdma-big.log sdma.log
	echo "change to test big mode"
fi
