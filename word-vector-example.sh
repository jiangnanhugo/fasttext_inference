#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"


make
#
#./fasttext skipgram -input "${DATADIR}"/fil9 -output "${RESULTDIR}"/fil9 -lr 0.025 -dim 100 \
#  -ws 5 -epoch 1 -minCount 5 -neg 5 -loss ns -bucket 500000 \
#  -minn 3 -maxn 6 -thread 12 -t 1e-4 -lrUpdateRate 100 -slim 0

./fasttext quantize -input  "${DATADIR}"/fil9 -output "${RESULTDIR}"/fil9 -qnorm -retrain -epoch 1 -slim 
#ut -f 1,2 "${DATADIR}"/rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > "${DATADIR}"/queries.txt

#at "${DATADIR}"/queries.txt | ./fasttext print-word-vectors "${RESULTDIR}"/fil9.bin > "${RESULTDIR}"/vectors.txt

#python eval.py -m "${RESULTDIR}"/vectors.txt -d "${DATADIR}"/rw/rw.txt
