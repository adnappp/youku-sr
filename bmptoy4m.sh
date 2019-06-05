#!/bin/bash

dir="./result_all_64"
name=0;
i=0

for file in `ls ${dir}`
do
    cd ${dir}
    if [ ${#file} -eq 17 ]; then   
        ffmpeg -i ${file}/1%3d.bmp  -pix_fmt yuv420p  -vsync 0 ${file}.y4m -y
    else 
	    ffmpeg -i ${file}/1%3d.bmp -vf select='not(mod(n\,25))' -pix_fmt yuv420p  -vsync 0 ${file}.y4m -y
	fi
    cd -
    i=$(( $i+1 ))
done;
