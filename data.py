import numpy as np

class Data:
    def __init__(self, filePath):
        self.filePath = filePath
        self.__LoadData()

    def __LoadData(self):
        print("Loading data...")
        data = np.genfromtxt("./DB_Collusion_All_processed.csv", delimiter=",", skip_header=1)

        # create dataset
        dataset = np.array([data[data[:, 0] == i][:, 1] for i in np.unique(data[:, 0])], dtype=object)
        labels = np.array([data[data[:, 0] == i][0, 4] for i in np.unique(data[:, 0])], dtype=object)
        country = np.array([data[data[:, 0] == i][0, -1] for i in np.unique(data[:, 0])], dtype=object)

        # create test and train data from shuffled dataset
        indices = np.arange(dataset.shape[0])
        np.random.shuffle(indices)

        trainIndices = indices[:int(0.8 * dataset.shape[0])]
        self.trainX = dataset[trainIndices]
        self.trainY = labels[trainIndices]

        testIndices = indices[int(0.8 * dataset.shape[0]):]
        self.testX = dataset[testIndices]
        self.testY = labels[testIndices]

        print("Data loaded")