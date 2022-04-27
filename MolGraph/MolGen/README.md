## Molecular fragment and Reassemble
Based on the eMolFrag repository， we develop a classical way to recombine the previously cleaved groups to generate new small molecules from mol2 structure format and and make some fixes for the previous bug in emolfrag to make it work properly.  

#### 1. Environment setup  
```ruby
tar -zxf openbabel-version.tar.gz # For example: *.tar.gz 
cd yourpath && mv openbabel-version openbabel
cd openbabel
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/apps/openbabel-2.3.1/
make -j 8 && make install -j 8
```
