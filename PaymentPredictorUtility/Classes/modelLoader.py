import pickle
import os
import time
from pathlib import Path


class modelLoader:

    def __init__(self, modelName, path, Folder=None, verbose=False):

        self.__Folder = None
        self.__model = None
        self.__Path = None
        __startTime = time.time()

        self.__Path = Path(path)

        if Folder is None:
            pass
        else:
            self.__Path = self.__Path.joinpath(str(Folder))

        self.__Path = self.__Path.joinpath(str(modelName))

        if verbose:
            self.__model = pickle.load(open(f"{self.__Path}", "rb"))
            print(f"Data set {modelName} loaded in {round(time.time() - __startTime, 2)} seconds")
        else:
            self.__model = pickle.load(open(f"{self.__Path}", "rb"))

    def getModel(self):
        return self.__model
