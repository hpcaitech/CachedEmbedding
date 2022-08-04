#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tbsm_py="python txt_to_npz.py "

$tbsm_py  --datatype="taobao" \
--num-train-pts=690000 --num-val-pts=300000 --points-per-user=10 
--numpy-rand-seed=123 --arch-embedding-size="987994-4162024-9439" 
--raw-train-file=./taobao_train.txt \
--raw-test-file=./taobao_test.txt \
--pro-train-file=./taobao_train_t20.npz \
--pro-val-file=./taobao_val_t20.npz \
--pro-test-file=./taobao_test_t20.npz \
--ts-length=20  