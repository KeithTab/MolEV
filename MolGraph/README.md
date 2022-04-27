## Main Use for code in this directory
Our main program for converting molecules to images is written in Go, where we compile it and integrate it with shell scripts to achieve batch output of images. Next, we will introduce the process and steps of operation.  

#### 1. Generate executable program rgb2bmp(Ubuntu20 system)  
First of all, you need to configure the relevant environment.  
```
sudo apt install gcc g++ gfortran build-essential zlib1g-dev libgsl-dev
```
