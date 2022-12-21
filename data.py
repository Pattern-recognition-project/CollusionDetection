import csv
from typing import List

class dataReader:
    def __init__(self, _filePath):
        """Data reader for a csv dataset

        Args:
            _filePath (string): path to dataset csv file. 
        """
        self.filePath = _filePath
        self.header = None

        self.dataset = open(self.filePath, newline='')
        self.datasetReader = csv.reader(self.dataset, delimiter=',')

    def getRows(self, hasHeader = True) -> List[List[str]]:
        """Extact rows from the dataset.

        Args:
            hasHeader (bool, optional): Set to true if dataset contains a header to extract the header to the header variable. Defaults to True.

        Returns:
            List[List[str]]: List of rows containing the information of the dataset.
        """
        rows = []
        for row in self.datasetReader:
            rows.append(row)
        
        if hasHeader: self.header = rows[0]
        return rows[1:]

    def __del__(self):
        self.dataset.close()



reader = dataReader("../collusion/DB_Collusion_All_processed.csv")
rows = reader.getRows()
print(reader.header)
print(rows[0])