// * description: create bmp file
// *******************************************************/

package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"unsafe"
//	"bufio"
//	"io"
//	"io/ioutil" 
)

// bmp RGB
type BitmapRGB struct {
	Blue          uint8
	Green         uint8
	Red           uint8
}

// bmp info header
type BitmapInfoHeader struct {
	Size           uint32;
	Width          int32;
	Height         int32;
	Places         uint16;
	BitCount       uint16;
	Compression    uint32;
	SizeImage      uint32;
	XperlsPerMeter int32;
	YperlsPerMeter int32;
	ClsrUsed       uint32;
	ClrImportant   uint32;
}

// bmp file header
type  BitmapFileHeader struct{
    Type uint16; 
} 

// bmp file header2
type  BitmapFileHeader2 struct{

    Size uint32;
    Reserved1 uint16;
    Reserved2 uint16;
    OffBits uint32;
} 

func check(e error) {
	if e != nil {
		panic(e)
	}
}

// check file is existed
func checkFileIsExist(filename string) bool {
	var exist = true
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		exist = false
	}
	return exist
}

func main() {

	//fmt.Println(argc)
	var index = 0;
	var fileName = "test2.bmp";
	var fileName2 = "test.bmp";
	var width = 1920;
	var height = 1080;
	var bit = 3;
	var red = 255;
	var green = 255;
	var blue = 255;
	var valueTmp = "12345";
	var strBlue = "255"
	var strGreen = "255"
	var strRed = "255"
	
	fmt.Println("main start, para info:");
	for idx, args := range os.Args{
		fmt.Println("para" + strconv.Itoa(idx) + ":", args);
		index = index + 1;

	}
	fmt.Println("index: " + strconv.Itoa(index))

	// check input commands
	if index < 8{
		fmt.Println("please input like this:");
		fmt.Println("./testbmp test.bmp 3 1920 1080 255 255 255");
		fmt.Println("test.bmp ------ bmp file name");
		fmt.Println("3        ------ 3 bytes RGB ");
		fmt.Println("1920     ------ bmp width ");
		fmt.Println("1080     ------ bmp height ");
		fmt.Println("255      ------ Blue ");
		fmt.Println("255      ------ Green ");
		fmt.Println("255      ------ Red ");
		return;
	}

	fileName2 = os.Args[1];
	valueTmp = os.Args[2]
	bit,err1 := strconv.Atoi(valueTmp)
	if(err1 != nil){
        fmt.Println("error1 happened ,exit")
        return  
    }
	width,err2 := strconv.Atoi(os.Args[3])
	if(err2 != nil){
        fmt.Println("error2 happened ,exit")
        return  
    }
    height,err3 := strconv.Atoi(os.Args[4])

	if(err3 != nil){
        fmt.Println("error3 happened ,exit")
        return  
    }
    blue,err4 := strconv.Atoi(os.Args[5])

	if(err4 != nil){
        fmt.Println("error4 happened ,exit")
        return  
    }
    green,err5 := strconv.Atoi(os.Args[6])

	if(err5 != nil){
        fmt.Println("error5 happened ,exit")
        return  
    }
    red,err6 := strconv.Atoi(os.Args[7])
	if(err6 != nil){
        fmt.Println("error6 happened ,exit")
        return 
    }

	strBlue = os.Args[5]
	strGreen = os.Args[6]
	strRed = os.Args[7]
	
	fmt.Println("fileName : " + fileName)
	fmt.Println("bit      : " + strconv.Itoa(bit))
	fmt.Println("width    : " + strconv.Itoa(width))
	fmt.Println("height   : " + strconv.Itoa(height))
	fmt.Println("blue     : " + strconv.Itoa(blue))
	fmt.Println("green    : " + strconv.Itoa(green))
	fmt.Println("red      : " + strconv.Itoa(red))
	fmt.Println("strBlue  : " + strBlue)
	fmt.Println("strGreen : " + strGreen)
	fmt.Println("strRed   : " + strRed)

	var err error;
	var file2 *os.File;

	/*
	var file *os.File;

	if checkFileIsExist(fileName){
		file, err = os.OpenFile(fileName, os.O_APPEND, 0666) //open file
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println("file is exist")
	} else {
		file, err = os.Create(fileName) //create file
		fmt.Println("file is not exist")
	}
	defer file.Close()

	//var headA, headB byte 
	//binary.Read(file, binary.LittleEndian, &headA)
	//binary.Read(file, binary.LittleEndian, &headB)
	binary.Read(file, binary.LittleEndian, &bmpFileHeader)

	fmt.Println(bmpFileHeader)
	
	//var size uint32
	//binary.Read(file, binary.LittleEndian, &size)
	binary.Read(file, binary.LittleEndian, &bmpFileHeader2)

	//var reservedA, reservedB uint16
	//binary.Read(file, binary.LittleEndian, &reservedA)
	//binary.Read(file, binary.LittleEndian, &reservedB)
	//binary.Read(file, binary.LittleEndian, &bmpFileHeader2.Reserved1)
	//binary.Read(file, binary.LittleEndian, &bmpFileHeader2.Reserved2)	

	//var offbits uint32
	//binary.Read(file, binary.LittleEndian, &offbits)
	//binary.Read(file, binary.LittleEndian, &bmpFileHeader2.OffBits)
	
	//fmt.Println(headA, headB, size, reservedA, reservedB, offbits)
	fmt.Println(bmpFileHeader2)
		
	
	infoHeader := new(BitmapInfoHeader)


	fmt.Println("infoHeader  size:", unsafe.Sizeof(infoHeader))
	fmt.Println("infoHeader2 size:", unsafe.Sizeof(infoHeader2))

	binary.Read(file, binary.LittleEndian, infoHeader)	
	binary.Read(file, binary.LittleEndian, &infoHeader2)
	fmt.Println(infoHeader)
	fmt.Println(infoHeader2)

	file.Sync();
	file.Close();
	//*/
	
	//*
	if checkFileIsExist(fileName2){
		file2, err = os.OpenFile(fileName2, os.O_APPEND, 0666) //open file
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println("file is exist")
	} else {
		file2, err = os.Create(fileName2) //create file
		fmt.Println("file is not exist")
	}
	defer file2.Close()
	//*/


	bmpFileHeader := BitmapFileHeader{};

	fmt.Println("bmpFileHeader  size:", unsafe.Sizeof(bmpFileHeader))

	bmpFileHeader2 := BitmapFileHeader2{};

	fmt.Println("bmpFileHeader2 size:", unsafe.Sizeof(bmpFileHeader2))

	infoHeader2 := BitmapInfoHeader{}


	bmpFileHeader.Type = 19778;

	bmpFileHeader2.Size = uint32(width) * uint32(height) * uint32(bit) + 54;//6220854;
	bmpFileHeader2.Reserved1 = 0;
	bmpFileHeader2.Reserved2 = 0;
	bmpFileHeader2.OffBits = 54;

	binary.Write(file2,binary.LittleEndian, bmpFileHeader) 

	binary.Write(file2,binary.LittleEndian, bmpFileHeader2) 




	
	infoHeader2.Size = uint32(40); // 

	fmt.Println("size    : " + strconv.Itoa(int(infoHeader2.Size)))
	
	infoHeader2.Width = int32(width);
	infoHeader2.Height = int32(height);
	infoHeader2.Places = 1;
	infoHeader2.BitCount = uint16(bit*8);
	infoHeader2.Compression = 0;
	infoHeader2.SizeImage = 0;
	infoHeader2.XperlsPerMeter = 3780;
	infoHeader2.YperlsPerMeter = 3780;
	infoHeader2.ClsrUsed = 0;
	infoHeader2.ClrImportant = 0;
	fmt.Println(infoHeader2)
	
	binary.Write(file2,binary.LittleEndian, infoHeader2) // write file



	//var d1 = []byte("123")
	//n2, err3 := file.Write(d1) // write file	
	//fmt.Println(n2);
	//binary.Write(file2,binary.LittleEndian, d1) 

	BitmapRGBData := BitmapRGB{uint8(blue),uint8(green),uint8(red)};
	fmt.Println(BitmapRGBData);
	
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			binary.Write(file2,binary.LittleEndian, BitmapRGBData) 
		}
	}
        
	file2.Sync();


	file2.Close();
}



