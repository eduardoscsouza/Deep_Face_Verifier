import numpy as np
import keras

class DataGenerator(keras.util.sequence):
    'Generates data for keras'
    
    def __init__(self, folder_list, batch_size=32, n_classes=2, length=1024, vector_length):
        self.folder_list = folder_list
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.length = length
        self.vector_length = vector_length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        batch = [np.empty((self.batch_size, self.vector_length)), np.empty((self.batch_size, self.vector_length))]
        Y = np.empty(self.batch_size)
        for i in range(self.batch_size):
            y = np.randint(0, 1)
            
            if y == 0:
                s1, s2 = np.random.choice(self.folder_list, 2)
                # vetor 1 de caracteristicas
                V1 = np.load('data/' + s1 + '/vector.npy')
                
                # vetor 2 de caracteristicas
                V2 = np.load('data/' + s2 + '/vector.npy')
                
                id1 = np.random.choice(V1.shape[0])
                V1 = V1[id1]
                
                id2 = np.random.choice(V2.shape[0])
                V2 = V2[id2]
                
                batch[0][i] = V1
                batch[1][i] = V2
                
                Y[i] = 0
                
            else:
                s = np.random.choice(self.folder_list)
                
                # vetor de caracteristicas
                V = np.load('data/' + s + '/vector.npy')
                
                id1, id2 = np.random.choice(V.shape[0], 2)
                
                V1 = V[id1]
                V2 = V[id2]
                
                batch[0][i] = V1
                batch[1][i] = V2
                Y[i] = 1
                
        return [batch1, batch2], Y