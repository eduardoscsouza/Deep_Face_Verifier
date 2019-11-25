#!/bin/bash

idx_filename="temp/CelebA/Anno/identity_CelebA_fix.txt"
imgs_dir="temp/CelebA/Img/img_align_celeba"
out_dir="dataset"

while read line
do
    in_img=$(printf "%s %s" $line | cut -d\  -f1)
    in_nam=$(printf "%s %s" $line | cut -d\  -f2)

    cur_indv=$out_dir"/celebA_"$in_nam
    mkdir -p $cur_indv

    orig=$imgs_dir"/"$in_img
    dest=$cur_indv"/"$in_img

    mv $orig $dest
done < $idx_filename