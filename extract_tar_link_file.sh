#!/bin/bash

mkdir dataset
tar -zxvf BSD_B_Centroid.tar.gz -C ./dataset/
unzip GOPRO_Large.zip -d ./dataset/
tar -zxvf RealBlur.tar.gz -C ./dataset/
tar -zxvf trained_model.tar.gz -C ./dataset/

sh link_file.sh