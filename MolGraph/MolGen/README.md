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
```ruby
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
## Run scripts to generate fragments of molecule  
```ruby
python ConfigurePath.py # In order to confirm your software path
python eMolFrag.py -i /path/to/your/mol2_dataset -o /path/to/results -p 2 -m 0 -c 0
```
#### 3. Result display  
<p align="center">
  <img alt="Light" src="https://github.com/CondaPereira/MolEV/blob/main/MolGraph/MolGen/img/group1.png" width="200">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Light" src="https://github.com/CondaPereira/MolEV/blob/main/MolGraph/MolGen/img/group4.png" width="200">
</p>  

## Generate creative molecule (not Orientation)  

```ruby
python /fragmix/fragmix.py --i your/path/to/emolfrag_output --draw --num your definition(default set to 10) 
```
