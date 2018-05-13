
#gather list of IDs
import keras
import numpy as np



class DataGenerator():

    

    def __init__(self, list_IDs, labels, batch_size=32, dim=(331,331), n_channels=3, 
        n_classes=228, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size  # Maybe a file that has the appropriate label mapping?
        self.list_IDs = list_IDs  # The ImageDataGenerator Instance
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Geenerate Data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #Generate Data
        for i, ID in enumerate(list_IDs_temp):
            #store sample
            X[i,] = np.load('processed_training_data/' + ID + '.npy')

            # Store Class
            y[i] = self.labels[ID]



