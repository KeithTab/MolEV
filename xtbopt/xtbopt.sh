#!/bin/bash

for ((i=692; i<=997; i++))
do
cd test_$i
xtb /mnt/c/Users/chris/Downloads/sdf/smi/test/test_$i/test_$i.sdf --opt extreme --charge 0 --alpb water  
cd ..
done

