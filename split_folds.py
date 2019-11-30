import os
from glob import glob
from queue import PriorityQueue
import numpy as np

dataset_dir = "autoencoder_dataset" # diretorio do data_set
n_splits = 5 # numero de splits 5 - fold cross validation

dirs = [cur_dir for cur_dir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cur_dir))] # pega todos os diretorios dentro do autoencoder_dataset
datasets = list(set([cur_dir.split("_")[0] for cur_dir in dirs])) # pega o cyberextruder, colorferet, etc.
datasets_indvs = [glob(os.path.join(dataset_dir, dataset+"_*")) for dataset in datasets] # cria um array [[(todos os caminhos dos individuos do cyberextruder)], [(todos os caminhos dos individuos do colorferet)], ...]

# Queremos que:
# - folds diferentes nao compartilhem mesmos individuos
# - quantidade de individuos diferentes em cada fold seja parecida
# - quantidade de imagens em cada fold seja parecida

folds = [[] for _ in range(n_splits)] # vai guardar os caminhos dos individuos de cada fold?
for dataset_indvs in datasets_indvs:
    indv_queue = PriorityQueue()
    for indv in dataset_indvs:
        indv_queue.put((-len(os.listdir(indv)) , indv)) # poe na priority queue com prioridade igual a quantidade de fotos que tem do individuo

    splits = [[] for _ in range(n_splits)]
    splits_queue = PriorityQueue()
    for i in range(n_splits):
        splits_queue.put((0, i)) # poe na priority queue com prioridade igual a quantidade de imagens no split

    max_size = len(dataset_indvs) // n_splits # maxima quantidade de individuos em um fold
    max_overflow = len(dataset_indvs) % n_splits # resto
    overflow_count = 0
    while not indv_queue.empty():
        top_indv_size, top_indv = indv_queue.get() # pega o individuo ainda em nenhum fold com maior quantidade de imagens
        cur_split_size, cur_split = splits_queue.get() # pega o split com menor quantidade de imagens

        top_indv_size *= -1 # pra voltar pro valor original
        cur_split_size += top_indv_size # adiciona quantidade de imagens do individuo escolhido
        splits[cur_split] += [top_indv] # append do individuo escolhido nesse split

        cur_split_len = len(splits[cur_split]) # ve quantidade de individuos nesse split

        # Essa parte garante que a quantidade de individuos vai ser dividida igualmente pra cada split
        if (cur_split_len < max_size): # se for menor que o maximo, poe de volta esse cara na queue
            splits_queue.put((cur_split_size, cur_split))
        elif (cur_split_len == max_size) and (overflow_count < max_overflow):
            splits_queue.put((cur_split_size, cur_split))
            overflow_count += 1

    sizes = [] # coloca quantidade total de imagens em cada split
    for split in splits:
        sizes += [np.sum([len(os.listdir(indv)) for indv in split])]
    print("Splits Subjects Counts:", [len(split) for split in splits]) # Printa numero de individuos de cada split
    print("Splits Images Counts:", sizes) # Printa numero de imagens em cada split
    print("----------------------")

    # adiciona os splits nos folds
    for i in range(n_splits):
        folds[i] += splits[i]

if __name__ == '__main__':
    # ordena cada fold dentro de folds
    folds = [sorted(fold) for fold in folds]

    # cria arquivos fold_1.txt, fold_2.txt, .. e printa neles os individuos em cada fold
    for i in range(n_splits):
        with open(os.path.join(dataset_dir, "fold_{}.txt".format(i)), "w") as fold_file:
            for indv in folds[i]:
                print(indv.split(os.sep)[-1], file=fold_file)