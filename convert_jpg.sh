#!/bin/bash

in_dir="dataset"
out_dir="temp_jpg"

find $in_dir -maxdepth 2 -mindepth 2 -type f | while read line
do
    cur_dir=$(printf %s $line | cut -d\/ -f2)
    mkdir -p $out_dir"/"$cur_dir

    cur_file=$(printf %s $line | cut -d\/ -f3)
    file_name=$(printf %s $cur_file | rev | cut -d\/ -f1 | cut -d. -f2 | rev)
    input_file=$in_dir"/"$cur_dir"/"$cur_file
    output_file=$out_dir"/"$cur_dir"/"$file_name".jpg"

    convert -quality 90 $input_file $output_file
done