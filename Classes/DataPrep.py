import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from Functions.Func import dateConvert


class DataCleaner:

    def __init__(self, inputFileName, ExportFileName, folder=None, path=None, verbose=False):

        self.__exportFileName = ExportFileName
        self.__folder = folder

        if path is not None:
            self.__Path = str(path) + "/"
        else:
            self.__Path = ""

        if folder is not None:
            self.df = pd.read_csv(f"{self.__Path}{folder}/{inputFileName}", low_memory=False)
        else:
            self.df = pd.read_csv(f"{self.__Path}{inputFileName}", low_memory=False)

        __timestart = time.time()
        __counter = 1

        if verbose:
            __sizeSave = self.df.shape
            self.__dataSelector('transaction_code', [204, 206])
            print(f"{abs(__sizeSave[0] - self.df.shape[0])} rows have been deleted in payment selection")
        else:
            self.__dataSelector('transaction_code', [204, 206])

        if verbose:
            __sizeSave = self.df.shape
            self.__dataSelector('transaction_description', ["P+I Principal Payment", "P+I Interest Payment"])
            print(f"{abs(__sizeSave[0] - self.df.shape[0])} rows have been deleted in payment cleaning")
        else:
            self.__dataSelector('transaction_description', ["P+I Principal Payment", "P+I Interest Payment"])

        if verbose:
            self.df = dateConvert(self.df)
            print("dates successfully converted from string to datetime format")
        else:
            self.df = dateConvert(self.df)

        if verbose:
            x = self.df.shape
            self.__nullColumnDropper(0.1)
            print(f"{abs(x[1] - self.df.shape[1])} columns have been deleted while dropping columns containing Null "
                  f"values")
        else:
            self.__nullColumnDropper(0.1)

        if verbose:
            __sizeSave = self.df.shape
            self.__nullRowDropper()
            print(f"{abs(__sizeSave[0] - self.df.shape[0])} rows have been deleted while dropping rows containing "
                  f"null values")
        else:
            self.__nullRowDropper()

        if verbose:
            self.__dataToCSV(self.df, self.__exportFileName, self.__folder, self.__Path)
            print("Cleaned Dataframe saved")
        else:
            self.__dataToCSV(self.df, self.__exportFileName, self.__folder, self.__Path)

    # Cleaning Functions
    def __dataSelector(self, columnName, valueList):
        self.df = self.df[(self.df[columnName].isin(valueList))]

    def __nullColumnDropper(self, threshold):
        dropList = []
        for col in self.df.columns:
            if self.df[col].isnull().sum() / len(self.df) >= threshold:
                dropList.append(col)
            else:
                pass

        self.df = self.df.drop(columns=dropList)

    def __nullRowDropper(self):
        self.df = self.df.dropna(axis=0).reset_index(drop=True)

    @staticmethod
    def __dataToCSV(dataframe, filename, folder=None, path=None):
        if folder is not None:
            dataframe.to_csv(f"{path}{folder}/{filename}")
        else:
            dataframe.to_csv(f"{path}{filename}")


class dataMLPrep:

    def __init__(self, inputFileName, exportDummyName, exportTargetName, folder=None, path=None, verbose=False):

        self.dummyDf = None
        self.targetDf = None

        if path is not None:
            self.__path = str(path) + "/"
        else:
            self.__path = ""

        if folder is not None:
            self.df = pd.read_csv(f"{self.__path}{folder}/{inputFileName}", index_col=0)
        else:
            self.df = pd.read_csv(f"{self.__path}{inputFileName}", index_col=0)

        self.__targetVariableCreator("paid")
        if verbose:
            print("target variable created")

        if verbose:
            x = self.df.shape
            self.__dataEncoder("paid")
            print(f"Data Labeling and Dummy Encoding resulted in {abs(x[1] - self.df.shape[1])} columns")
        else:
            self.__dataEncoder("paid")

        self.__dataToCSV(self.dummyDf, exportDummyName, folder, path=self.__path)
        if verbose:
            print("Dummy Dataframe saved")

        self.__dataToCSV(self.targetDf, exportTargetName, folder, path=self.__path)
        if verbose:
            print("Target Dataframe saved")

    # Functions
    def __dataEncoder(self, targetVar):
        labelEncoder = LabelEncoder()
        colList = list(self.df.select_dtypes(include=['object']).columns)
        for col in colList:
            self.df[col] = labelEncoder.fit_transform(self.df[col])

        # Remove target variable
        dfTarget = self.df[str(targetVar)]
        self.df.drop(targetVar, axis=1, inplace=True)

        # Dummy encode
        dummyDf = pd.get_dummies(self.df)

        self.dummyDf = dummyDf
        self.targetDf = dfTarget

    def __targetVariableCreator(self, targetName):
        target_list = []
        for row in range(len(self.df)):
            if self.df.date_due[row] <= self.df.transaction_date[row]:
                target_list.append(1)
            else:
                target_list.append(0)

        self.df = self.df.drop(['date_due', 'transaction_date'], axis=1)

        self.df[targetName] = target_list

    @staticmethod
    def __dataToCSV(dataframe, filename, folder=None, path=None):
        if folder is not None:
            dataframe.to_csv(f"{path}{folder}/{filename}")
        else:
            dataframe.to_csv(f"{path}{filename}")
