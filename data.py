import numpy as np
import pandas as pd

from scipy.stats import moment, skew, kurtosis, variation
from collections import Counter

class Data:
    def __init__(self, filePath):
        self.filePath = filePath
        self.__LoadData()

    def __LoadData(self):
        print("Loading data...")
        data = np.genfromtxt("./DB_Collusion_All_processed.csv", delimiter=",", skip_header=1)

        # create dataset
        self.dataset = np.array([data[data[:, 0] == i][:, 1] for i in np.unique(data[:, 0])], dtype=object)
        self.labels = np.array([data[data[:, 0] == i][0, 4] for i in np.unique(data[:, 0])], dtype=object)
        self.country = np.array([data[data[:, 0] == i][0, -1] for i in np.unique(data[:, 0])], dtype=int)

        # create test and train data from shuffled dataset
        self.indices = np.arange(self.dataset.shape[0])
        np.random.shuffle(self.indices)

        # trainIndices = indices[:int(0.8 * self.dataset.shape[0])]
        # self.trainX = dataset[trainIndices]
        # self.trainY = labels[trainIndices]

        # testIndices = indices[int(0.8 * dataset.shape[0]):]
        # self.testX = dataset[testIndices]
        # self.testY = labels[testIndices]

        print("Data loaded")


    def __get_data(kind, indices):

        if kind == 'data':
            return self.dataset[indices]
        else:
            return self.labels[indices]

    def get_train_X():
        return self.__get_data('data', self.indices[:int(0.8 * self.dataset.shape[0])])

    def get_train_y():
        return self.__get_data('labels', self.indices[:int(0.8 * self.dataset.shape[0])])

    def get_test_X():
        return self.__get_data('data', self.indices[int(0.8 * self.dataset.shape[0]):])

    def get_test_y():
        return self.__get_data('labels', self.indices[int(0.8 * self.dataset.shape[0]):])


    def __proportion(self, arr):

        arr = sum([list(str(num)) for num in arr], [])

        counts = Counter(arr)

        x = [counts.get(str(i), 0) for i in range(10)]

        return x

    def __country(self):
        """
        Creates one hot encoded columns of the country
        
        """
        print(self.country[:4])
        n_values = np.max(self.country) + 1
        return np.eye(n_values)[self.country] 

    def __average_auction(self):

        return self.country == 1
    
    def __winner_auction(self):

        return self.country != 1


    def load_aggegrated(self, data_type='numpy'):

        labels = [
            'num_bids', 'mean', 'variance', 'skew', 'kurtosis',
            'variation', 'min',
            'max', *[f"proportion_{i}" for i in range(10)],
            *[f"country_{i}" for i in np.unique(self.country)],
            'first_price_auction', 'average_price_auction'
        ]

        print(self.dataset[:4])

        agg_data = np.stack(
            [
                [len(x) for x in self.dataset],
                [np.mean(x) for x in self.dataset],
                [np.var(x) for x in self.dataset],
                [skew(x) for x in self.dataset],
                [kurtosis(x) for x in self.dataset],
                [variation(x) for x in self.dataset],
                [min(x) for x in self.dataset],
                [max(x) for x in self.dataset],
                *np.array([self.__proportion(x) for x in self.dataset]).transpose(),
                *self.__country().transpose(),
                self.__winner_auction(),
                self.__average_auction()
            ],
            axis=1
        )

        if data_type == 'numpy':
            return agg_data
        elif data_type == 'pandas':
            return pd.DataFrame(agg_data, columns=labels)
        elif data_type == 'dict':
            return pd.DataFrame(agg_data, columns=labels).to_dict('records')


if __name__ == "__main__":

    data = Data("./DB_Collusion_All_processed.csv")

    agg_data = data.load_aggegrated(data_type='pandas')


    print(agg_data)

    agg_data.to_csv('test.csv')



        

