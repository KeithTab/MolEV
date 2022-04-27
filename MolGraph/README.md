## Main Use for code in this directory
Our main program for converting molecules to images is written in Go, where we compile it and integrate it with shell scripts to achieve batch output of images. Next, we will introduce the process and steps of operation.  

#### 1. Generate executable program rgb2bmp(Ubuntu20 system)  
First of all, you need to configure the relevant environment.  
```
sudo apt update && sudo apt upgrade -y
sudo apt install gcc g++ gfortran build-essential zlib1g-dev libgsl-dev
sudo apt install golang(go_version >= 1.14)
conda create -n rdkit python=3.8
conda activate rdkit 
conda conda install -c conda-forge rdkit
```
Secondly, the python dependencies you need.  
#### 2. Python Dependen## Setup and dependencies 

Dependencies:
- argparse
- rdkit >= 2020.09.3.0
- numpy 
- pillow 
- pandas
- pennylane 
- contexlib >= 21.6.0
- multiprocessing

#### 3. Compile the exec transfer
```
go build rgb2bmp.go (-o) your filename
export PATH=$PATH:/yourpathtoexec
```
you can try to run a test as follows:
```
rgb2bmp test.bmp 3 1920 1080 255 255 0
```
if it have output as follows, you have been access to success:
```ruby
main start, para info:
para0: testbmp
para1: test.bmp
para2: 3
para3: 1920
para4: 1080
para5: 255
para6: 255
para7: 0
index: 8
fileName : test2.bmp
bit      : 3
width    : 1920
height   : 1080
blue     : 255
green    : 255
red      : 0
strBlue  : 255
strGreen : 255
strRed   : 0
file is not exist
bmpFileHeader  size: 2
bmpFileHeader2 size: 12
size    : 40
{40 1920 1080 1 24 0 0 3780 3780 0 0}
{255 255 0}
```
