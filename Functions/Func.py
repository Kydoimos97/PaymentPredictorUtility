import os
from pathlib import Path

import pandas as pd


def dateConvert(dataframe):
    for colName in dataframe:
        try:
            if "date" in colName or "next" in colName:
                dataframe[colName] = pd.to_datetime(dataframe[colName])
            else:
                pass
        except TypeError:
            pass

    return dataframe


def directoryScanner(Name, SourcePath, Folder=None, returnMethod="Last"):
    if Folder is None:
        path = Path(SourcePath)
    else:
        path = Path(SourcePath).joinpath(Folder)

    scanObj = os.scandir(path)
    returnCTime = None
    for file in scanObj:
        if str(Name).lower() in file.name.lower():
            if returnMethod.lower() == "last":
                if returnCTime is None:
                    returnCTime = [file.name, float(file.stat().st_ctime)]
                elif returnCTime[1] < float(file.stat().st_ctime):
                    returnCTime = [file.name, float(file.stat().st_ctime)]
                else:
                    pass
            if returnMethod.lower() == "first":
                if returnCTime is None:
                    returnCTime = [file.name, float(file.stat().st_ctime)]
                elif returnCTime[1] > float(file.stat().st_ctime):
                    returnCTime = [file.name, float(file.stat().st_ctime)]
                else:
                    pass
    return str(returnCTime[0])
