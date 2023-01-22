import numpy as np
import pandas as pd
import pickle

from scipy.stats import moment, skew, kurtosis, variation
from collections import Counter

class Data:
    def __init__(self, filePath):
        self.filePath = filePath
        self.__LoadData()

    def __LoadData(self):
        print("Loading data...")

        data = np.genfromtxt(self.filePath, delimiter=",", skip_header=1)

        # create dataset
        self.dataset = np.array([data[data[:, 0] == i][:, 1] for i in np.unique(data[:, 0])], dtype=object)
        self.ogdataset = np.array([data[data[:, 0] == i][:, -1] for i in np.unique(data[:, 0])], dtype=object)
        self.labels = np.array([data[data[:, 0] == i][0, 4] for i in np.unique(data[:, 0])], dtype=object)
        self.country = np.array([data[data[:, 0] == i][0, -2] for i in np.unique(data[:, 0])], dtype=int)

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

    def __get_data(self, kind, indices):

        if kind == 'data':
            return self.dataset[indices]
        else:
            return self.labels[indices]

    def get_train_X(self):
        return self.__get_data('data', self.indices[:int(0.8 * self.dataset.shape[0])])

    def get_train_y(self):
        return self.__get_data('labels', self.indices[:int(0.8 * self.dataset.shape[0])])

    def get_test_X(self):
        return self.__get_data('data', self.indices[int(0.8 * self.dataset.shape[0]):])

    def get_test_y(self):
        return self.__get_data('labels', self.indices[int(0.8 * self.dataset.shape[0]):])


    def __proportion(self, arr):
        if np.isnan(arr[0]):
            # the original bid values could not be found
            return [0 for i in range(10)]

        arr = sum([list(str(num)) for num in arr], [])
        counts = Counter(arr)
        total_digits = np.sum([counts.get(str(i),0) for i in range(10)])
        return [counts.get(str(i), 0)/total_digits for i in range(10)]

    def __country(self):
        """
        Creates one hot encoded columns of the country

        """

        n_values = np.max(self.country) + 1
        return np.eye(n_values)[self.country]

    def __average_auction(self):

        return self.country == 1

    def __winner_auction(self):

        return self.country != 1



    def get_margin(self, auction, difference):

        auction = np.sort(auction)[::-1]
        return auction[0] - auction[difference]



    def get_margins_per_country(self):

        countries = ['Brazil','Italy','America','Switzerland_GR_SG','Switzerland_Ticino','Japan']

        results = []
        for i in range(6):

            country_data = self.dataset[self.country == i]

            margin1 = np.mean([self.get_margin(x, 1) for x in country_data if len(x) >= 2])
            margin2 = np.mean([self.get_margin(x, 2) for x in country_data if len(x) >= 3])

            results.append(
                {
                    'name': countries[i],
                    'margin1': np.around(margin1, decimals=3),
                    'margin2': np.around(margin2, decimals=3),
                }
            )

        return pd.DataFrame(results)

    def get_agg_train(self):

        agg_data = self.load_aggegrated()

        return agg_data[self.indices[:int(0.8 * agg_data.shape[0])]]

    def get_agg_test(self):

        agg_data = self.load_aggegrated()

        return agg_data[self.indices[int(0.8 * agg_data.shape[0]):]]


    def load_aggegrated(self, data_type='numpy', add_labels=False, min_bids=1):


        countries = ['Brazil','Italy','America','Switzerland_GR_SG','Switzerland_Ticino','Japan']
        columns = [
            'num_bids', 'mean', 'variance', 'skew', 'kurtosis',
            'variation', 'min',
            'max', *[f"proportion_{i}" for i in range(10)],
            *[f"is_{countries[i]}" for i in np.unique(self.country)],
            'first_price_auction', 'average_price_auction'
        ]

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
                *np.array([self.__proportion(x) for x in self.ogdataset]).transpose(),
                *self.__country().transpose(),
                self.__winner_auction(),
                self.__average_auction()
            ],
            axis=1
        )


        if add_labels:

            columns = ['labels'] + columns
            agg_data = np.stack(
                [
                    self.labels,
                    *(agg_data.transpose()),
                ],
                axis=1
            )

        selection = [x >= min_bids for x in agg_data[:,columns.index('num_bids')]]
        agg_data = agg_data[selection,:]


        if data_type == 'numpy':
            return agg_data
        elif data_type == 'pandas':
            return pd.DataFrame(agg_data, columns=columns)
        elif data_type == 'dict':
            return pd.DataFrame(agg_data, columns=columns).to_dict('records')


if __name__ == "__main__":

    df = Data("./DB_Collusion_All_processed.csv")

    agg_data = df.load_aggegrated(data_type='pandas', add_labels=True, min_bids=1)

    # print(df.get_test_X())


    # print(agg_data)

    # agg_data.to_csv('test.csv')

    # with open("DB_Collusion_All_processed.obj","wb") as filehandler:
    #     pickle.dump(df, filehandler)

    # with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    #     df = pickle.load(filehandler)

    # margins = data.get_margins_per_country()
    # print(margins)
