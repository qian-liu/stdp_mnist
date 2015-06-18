#!/bin/bash
for i in `seq 200 200 1999`;
do
    python training.py $i
done   
