# MolEV
[![Header](https://github.com/CondaPereira/MolEV/blob/main/images/Molecular.png "Header")](https://some-url.dev/)
--------------------------------------------------------------------------------
As follow is the work of team VE_CPU-iDEC this year:

## Micromolecular Model Setup
Our model focuses on data extraction of three sets of data: the sum of bond lengths of molecules, the angle between two atoms, and the diagonal of parallelograms. The molecular information mentioned above is all in three dimensions, which can improve our docking model to some extent. In addition, the molecular 3D model we rendered is supported by blender.After counting these three sets of data we will convert them into HSV plots by a certain formula instead of RGB, and according to a lot of research, HSV is better than RGB in data science applications.  

![Model_setup](https://github.com/CondaPereira/MolEV/blob/main/images/Model_1.png)

### Way to optimize our micromolcules
We will develop a deep learning based molecule generation framework and compare it with the optimized small molecule structures from the other two approaches and measure the accuracy of our model by this criterion, now we have mainly optimized the small molecule bulk structure of xtb at GFN2-xtb level and bulk small molecule optimization under the MMFF force field of RDkit, the model ConfEVG is still in the development stage and will be updated at a later stage.

## Protein Model Setup
The protein-based model is still under development, and the dataset we use is mainly from PDBbind and Uniprot, after which we will also extract the 3D information from the dataset for the target information as input to our model.  
PDBbind link: http://www.pdbbind.org.cn/quickpdb.php?quickpdb=5ho7  
Uniprot link: https://www.uniprot.org/uniprot/?query=Human&sort=score  

## Work Timeline(Updating)
