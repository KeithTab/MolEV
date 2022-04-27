## Molecular fragment and Reassemble
Based on the eMolFrag repository， we develop a classical way to recombine the previously cleaved groups to generate new small molecules from mol2 structure format and and make some fixes for the previous bug in emolfrag to make it work properly.  

## Environment setup  
#### 1. Openbabel build process
```ruby
tar -zxf openbabel-version.tar.gz 
cd yourpath && mv openbabel-version openbabel
cd openbabel
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/apps/openbabel-2.3.1/
make -j 8 && make install -j 8
export PATH=$PATH:/path/to/openbabel/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/openbabel/lib
```
#### 2. eSynth build process
```
tar -zxf esynth-version.gz
mv esynth-version esynth
cd esynth
vim Makefile just as follows
OB_INC= /path/to/openbabel/include/openbabel-2.0/

OB_LIB= /path/to/openbabel/lib/

GSL_LIB=/usr/lib64/
GSL_INC=/usr/include/
make
export PATH=$PATH:/path/to/esynth/src
```
