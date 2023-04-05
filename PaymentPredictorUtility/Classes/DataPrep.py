#  Copyright (C) 2022-2023 - Willem van der Schans - All Rights Reserved.
#
#  THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL.
#  UNAUTHORIZED COPYING, TRANSFERRING OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM IS STRICTLY PROHIBITED.
#  The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them
#  for any purpose other than the purpose for which they were provided to you.
#
#  The software is provided "AS IS", without warranty of any kind, express or implied, including but not limited to
#  the warranties of merchantability, fitness for a particular purpose and non infringement.
#  In no event shall the authors or copyright holders be liable for any claim, damages or other liability,
#  whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software
#  or the use or other dealings in the software.
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from PaymentPredictorUtility.Functions.Func import dateConvert


class DataCleaner:

    def __init__(self, inputFileName, ExportFileName, folder=None, path=None, verbose=False):

        """
    The __init__ function is the first function that runs when an object of this class is created.
    It takes in a file name, and exports a cleaned version of the dataframe to another csv file.
    The __init__ function also has optional parameters for folder and path, which are used if you want to specify where
    the inputFileName is located or where you want your output CSV file to be saved.

    Args:
        self: Represent the instance of the class
        inputFileName: Specify the name of the file to be cleaned
        ExportFileName: Specify the name of the file that will be exported
        folder: Specify the folder in which the input file is located
        path: Specify the path to the folder containing the data
        verbose: Print the number of rows and columns deleted at each step

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
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

   
    def __dataSelector(self, columnName, valueList):
        """
    The __dataSelector function takes in a column name and a list of values.
    It then filters the dataframe to only include rows where the value in that column is one of those values.

    Args:
        self: Allow an object to refer to itself inside of a method
        columnName: Select a column in the dataframe
        valueList: Select the values that are in the columnname parameter

    Returns:
        A dataframe with the columnname and valuelist specified

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.df = self.df[(self.df[columnName].isin(valueList))]

    def __nullColumnDropper(self, threshold):
        """
    The __nullColumnDropper function takes a threshold as an argument and drops columns that have more than the
    threshold percentage of null values. The function is called by the dropNullColumns method, which takes a threshold
    as an argument and calls __nullColumnDropper with that value.

    Args:
        self: Make the function a method of the class
        threshold: Determine the percentage of null values in a column that would trigger it to be dropped

    Returns:
        A list of columns that are to be dropped

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        dropList = []
        for col in self.df.columns:
            if self.df[col].isnull().sum() / len(self.df) >= threshold:
                dropList.append(col)
            else:
                pass

        self.df = self.df.drop(columns=dropList)

    def __nullRowDropper(self):
        """
    The __nullRowDropper function drops all rows that contain null values.
        It is called by the __nullColumnDropper function, which in turn is called by the cleanData method.

    Args:
        self: Allow an object to refer to itself inside of a method

    Returns:
        A dataframe with all rows that contain null values removed

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.df = self.df.dropna(axis=0).reset_index(drop=True)

    @staticmethod
    def __dataToCSV(dataframe, filename, folder=None, path=None):
        """
    The __dataToCSV function takes in a dataframe, filename, folder (optional), and path (optional)
    and saves the dataframe as a CSV file. If no folder is specified, it will save to the current working directory.
    If no path is specified, it will save to the current working directory.

    Args:
        dataframe: Pass the dataframe into the function
        filename: Specify the name of the file that will be created
        folder: Specify a folder to save the file in
        path: Specify the path to the folder where you want to save your file

    Returns:
        A csv file

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if folder is not None:
            dataframe.to_csv(f"{path}{folder}/{filename}")
        else:
            dataframe.to_csv(f"{path}{filename}")


class dataMLPrep:

    def __init__(self, inputFileName, exportDummyName, exportTargetName, folder=None, path=None, verbose=False):

        """
    The __init__ function is the first function that gets called when an object of this class is created.
    It takes in a file name, and two export names for the dummy dataframe and target dataframe respectively.
    The folder parameter allows you to specify which folder your input file is located in, if it's not in the same directory as this script.
    The path parameter allows you to specify where your output files will be saved (if different from where they are being read).
    If verbose=True then some print statements will appear during execution.

    Args:
        self: Refer to the object itself
        inputFileName: Specify the name of the file that will be read in
        exportDummyName: Specify the name of the file that will be created to store dummy data
        exportTargetName: Name the target dataframe
        folder: Specify the folder where the input file is located
        path: Set the path of the folder where all files are located
        verbose: Print out the progress of the function

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
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

   
    def __dataEncoder(self, targetVar):
        """
    The __dataEncoder function takes in a target variable and performs the following steps:
        1. Label encodes all categorical variables
        2. Removes the target variable from the dataframe
        3. Dummy encodes all categorical variables (including those that were label encoded)

    Args:
        self: Refer to the class itself
        targetVar: Identify the target variable in the dataframe

    Returns:
        A dummy encoded dataframe and a target variable

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        labelEncoder = LabelEncoder()
        colList = list(self.df.select_dtypes(include=['object']).columns)
        for col in colList:
            self.df[col] = labelEncoder.fit_transform(self.df[col])

       
        dfTarget = self.df[str(targetVar)]
        self.df.drop(targetVar, axis=1, inplace=True)

       
        dummyDf = pd.get_dummies(self.df)

        self.dummyDf = dummyDf
        self.targetDf = dfTarget

    def __targetVariableCreator(self, targetName):
        """
    The __targetVariableCreator function takes in a targetName and creates a new column in the dataframe with that name.
    The values of this column are determined by comparing the date_due and transaction_date columns. If date_due is less than or equal to
    transaction_date, then 1 is appended to the list, otherwise 0 is appended to the list. The function then drops both date columns from
    the dataframe and adds on our newly created target variable.

    Args:
        self: Allow an object to refer to itself inside of a method
        targetName: Name the target variable column

    Returns:
        A list of 1's and 0's

    Doc Author:
        Willem van der Schans, Trelent AI
    """
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
        """
    The __dataToCSV function takes in a dataframe, filename, folder (optional), and path (optional)
    and saves the dataframe as a CSV file. If no folder is specified, it will save to the current working directory.
    If no path is specified, it will save to the current working directory.

    Args:
        dataframe: Store the dataframe that is to be saved as a csv file
        filename: Name the file that is being created
        folder: Specify the folder where the file will be saved
        path: Specify the path to where you want to save your file

    Returns:
        A csv file in the specified folder

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if folder is not None:
            dataframe.to_csv(f"{path}{folder}/{filename}")
        else:
            dataframe.to_csv(f"{path}{filename}")
