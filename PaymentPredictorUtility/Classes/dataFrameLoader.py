import os
import time
from pathlib import Path

import pandas as pd


class dataFrameLoader:

    def __init__(self, csvName, path, Folder=None, verbose=False):

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
        return self.__df
