#!/bin/bash

dir="./temp"
name=0;
for file in `ls ${dir}`
do
    name=${file%.*}
    cd ${dir}
    mkdir ${name}
	ffmpeg -i ${name}.y4m -vsync 0 ${name}/%3d.bmp -y
    cd -
done;
