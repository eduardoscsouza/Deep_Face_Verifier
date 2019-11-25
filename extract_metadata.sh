#!/bin/bash

imdb_out_filename="imdb_crop/imdb_idxs.txt"
wiki_out_filename="wiki_crop/wiki_idxs.txt"
imdb_imgs_dir="imdb_crop"
wiki_imgs_dir="wiki_crop"
out_dir="imdb_wiki_sep"

count=0
while read line
do
    in_file=$(printf "%s\t%s" $line | cut -d$'\t' -f1)
    in_name=$(printf "%s\t%s" $line | cut -d$'\t' -f2)

    cur_indv=$out_dir"/imdb_"$in_name
    mkdir -p $cur_indv

    cp $imdb_imgs_dir"/"$in_file $cur_indv"/"$count".jpg"
    ((count++))
done < $imdb_out_filename

while read line
do
    in_file=$(printf "%s\t%s" $line | cut -d$'\t' -f1)
    in_name=$(printf "%s\t%s" $line | cut -d$'\t' -f2)

    cur_indv=$out_dir"/wiki_"$in_name
    mkdir -p $cur_indv

    cp $wiki_imgs_dir"/"$in_file $cur_indv"/"$count".jpg"
    ((count++))
done < $wiki_out_filename