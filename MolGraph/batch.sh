#!/bin/bash

for ((i=1; i<=996; i++));
do
	mv 'geomparm out '$i.txt test_$i.txt
	mkdir test_$i
	mv test_$i.txt /home/szk/sdf/txt/test_$i/test_$i.txt
	cd test_$i
	awk '{print $1,$2,$3}' /home/szk/sdf/txt/test_$i/test_$i.txt > real_$i.txt
	cd ..
	cd test_$i
	awk '{print $1}' real_$i.txt > list1.txt
	awk '{print $2}' real_$i.txt > list2.txt
	awk '{print $3}' real_$i.txt > list3.txt
	cd ..
	/home/szk/miniconda3/bin/python normalizer.py -i /home/szk/sdf/txt/test_$i/list1.txt -o /home/szk/sdf/txt/test_$i/1.txt
	/home/szk/miniconda3/bin/python normalizer.py -i /home/szk/sdf/txt/test_$i/list2.txt -o /home/szk/sdf/txt/test_$i/2.txt
	/home/szk/miniconda3/bin/python normalizer.py -i /home/szk/sdf/txt/test_$i/list3.txt -o /home/szk/sdf/txt/test_$i/3.txt
	cd test_$i
	while read number
	do 
		echo -n "0x"
		echo "obase=16; ibase=10; $number" | bc 
	done < 1.txt > a1.txt
	while read number
	do 
		echo -n "0x" 
		echo "obase=16; ibase=10; $number" | bc 
	done < 2.txt > a2.txt
	while read number
	do 
		echo -n "0x"
		echo "obase=16; ibase=10; $number" | bc 
	done < 3.txt > a3.txt
	paste a1.txt a2.txt a3.txt >> all.txt
	paste 1.txt 2.txt 3.txt >> dec.txt
	cd ..
done

for ((j=1; j<=996; j++));
do
	cd test_$j
	n=`grep -c "" dec.txt`
	for ((k=1; k<=n; k++));
	do
		head -$k dec.txt | tail -n +$k > rgb_temp.txt
		echo "rgb2bmp go.bmp 3 300 300" > command_front.txt
		paste command_front.txt rgb_temp.txt > command.sh && chmod +x command.sh && ./command.sh && mv go.bmp go_$k.bmp
		rm -rf rgb_temp.txt
		rm -rf command.sh
	done
	rm -rf *.txt
	cd ..
	/home/szk/miniconda3/bin/python /home/szk/sdf/txt/merge.py -i /home/szk/sdf/txt/test_$j
	rm -rf test_$j
done

