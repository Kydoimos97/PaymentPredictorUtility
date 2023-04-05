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

import pickle
import os
import time
from pathlib import Path


class modelLoader:

    def __init__(self, modelName, path, Folder=None, verbose=False):

        """
    The __init__ function is the constructor for a class. It is called when an object of that class
    is instantiated. The __init__ function can take any number of arguments, but it must have at least one argument,
    self (which refers to the newly created object). All other arguments are passed as parameters to the __init__ function.

    Args:
        self: Represent the instance of the class
        modelName: Specify the name of the model that is being loaded
        path: Specify the path to the folder where we want to store our model
        Folder: Specify the folder in which the model is located
        verbose: Print out the time it takes to load a model

    Returns:
        Nothing

    Doc Author:
        Trelent
    """
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
        """
    The getModel function returns the model.

    Args:
        self: Represent the instance of the class

    Returns:
        The value of the __model attribute

    Doc Author:
        Trelent
    """
        return self.__model
