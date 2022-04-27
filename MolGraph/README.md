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
#### 2. Setup Python dependencies 

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
When you have finished generating one color image, you will need to merge the images of whole molecule so that you can get a color map about one molecular for classification, meanwhile we plan to calculate the AQED, containing molecular descriptors such as the number of benzene rings for the subsequent image classification problem. If you only want to generate one group data based on (θ,δ,φ), you can infer the C++ code 'main.cpp':  
```ruby
int main(const int argc, const char** argv)
{
    int  width = 640;
    int  height = 360;
    int  format = FORMAT_RGB;
    
    char colorR = 0xff;
    char colorG = 0xff;
    char colorB = 0x00;
    
    int size = width * height * format;
    char* pRgb = (char*)malloc(size);

    for (int i = 0; i < width * height; i++)
    {
        pRgb[i * format] = colorR[1024];
        pRgb[i * format + 1] = colorG;
        pRgb[i * format + 2] = colorB;
    }

    rgbaToBmpFile((char*)"test.bmp", pRgb, width, height, format);

    free(pRgb);
}
```
When you need to merge the images to a color map, please run steps as follows:
```
1. chmod +x batch.sh
2. ./batch.sh
```

## Results  
<center class="half">
    <img src="https://github.com/CondaPereira/MolEV/blob/main/MolGraph/img/test_1.bmp" width="200" style="display: inline-block" /><img src="https://github.com/CondaPereira/MolEV/blob/main/MolGraph/img/test_2.bmp" width="200" style="display: inline-block" /><img src="https://github.com/CondaPereira/MolEV/blob/main/MolGraph/img/test_3.bmp" width="200" style="display: inline-block" /> </center>
