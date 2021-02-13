#!/bin/bash
for i in 0 1 2 3 4
do
python train.py --fold $i --model rf
done
