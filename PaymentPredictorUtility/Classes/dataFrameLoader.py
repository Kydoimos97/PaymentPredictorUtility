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

import os
import time
from pathlib import Path

import pandas as pd


class dataFrameLoader:

    def __init__(self, csvName, path, Folder=None, verbose=False):

        """
    The __init__ function is the first function that gets called when you create a new instance of a class.
    It's job is to initialize all of the attributes of the newly created object.


    Args:
        self: Refer to the object itself
        csvName: Specify the name of the csv file to be loaded
        path: Set the path to the folder where the csv file is located
        Folder: Specify the folder in which the csv file is located
        verbose: Print out the time it takes to load a data set

    Returns:
        Nothing

    Doc Author:
        Trelent
    """
        self.__Folder = None
        self.__Path = None
        self.__df = None
        __startTime = time.time()

        self.__Path = Path(path)

        if Folder is None:
            pass
        else:
            self.__Path = self.__Path.joinpath(str(Folder))

        self.__Path = self.__Path.joinpath(str(csvName))

        if verbose:
            self.__df = pd.read_csv(f"{self.__Path}", index_col=0, engine="pyarrow")
            print(f"Data set {csvName} loaded in {round(time.time() - __startTime, 2)} seconds")
        else:
            self.__df = pd.read_csv(f"{self.__Path}", index_col=0, engine="pyarrow")

    def getDf(self):
        """
    The getDf function returns the dataframe that was created in the __init__ function.


    Args:
        self: Represent the instance of the class

    Returns:
        The dataframe that was created in the constructor

    Doc Author:
        Trelent
    """
        return self.__df
