import os
from glob import glob
from queue import PriorityQueue
import numpy as np

dataset_dir = "autoencoder_dataset"
n_splits = 5

dirs = [cur_dir for cur_dir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cur_dir))]
datasets = list(set([cur_dir.split("_")[0] for cur_dir in dirs]))
datasets_indvs = [glob(os.path.join(dataset_dir, dataset+"_*")) for dataset in datasets]

folds = [[] for _ in range(n_splits)]
for dataset_indvs in datasets_indvs:
    indv_queue = PriorityQueue()
    for indv in dataset_indvs:
        indv_queue.put((-len(os.listdir(indv)) , indv))

    splits = [[] for _ in range(n_splits)]
    splits_queue = PriorityQueue()
    for i in range(n_splits):
        splits_queue.put((0, i))

    max_size = len(dataset_indvs) // n_splits
    max_overflow = len(dataset_indvs) % n_splits
    overflow_count = 0
    while not indv_queue.empty():
        top_indv_size, top_indv = indv_queue.get()
        cur_split_size, cur_split = splits_queue.get()

        top_indv_size *= -1
        cur_split_size += top_indv_size
        splits[cur_split] += [top_indv]

        cur_split_len = len(splits[cur_split])
        if (cur_split_len < max_size):
            splits_queue.put((cur_split_size, cur_split))
        elif (cur_split_len == max_size) and (overflow_count < max_overflow):
            splits_queue.put((cur_split_size, cur_split))
            overflow_count += 1

    sizes = []
    for split in splits:
        sizes += [np.sum([len(os.listdir(indv)) for indv in split])]
    print("Splits Subjects Counts:", [len(split) for split in splits])
    print("Splits Images Counts:", sizes)
    print("----------------------")

    for i in range(n_splits):
        folds[i] += splits[i]

folds = [sorted(fold) for fold in folds]
for i in range(n_splits):
    with open(os.path.join(dataset_dir, "fold_{}.txt".format(i)), "w") as fold_file:
        for indv in folds[i]:
            print(indv.split(os.sep)[-1], file=fold_file)