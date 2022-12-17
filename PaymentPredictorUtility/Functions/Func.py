import os
from pathlib import Path
from colorama import Style, Fore, init
import pandas as pd
from datetime import datetime

init(autoreset=True)


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


def directoryScanner(Name, SourcePath, Folder=None, returnMethod="Last", loadingAnimator=None, skipFlag=False):
    if Folder is None:
        path = Path(SourcePath)
    else:
        path = Path(SourcePath).joinpath(Folder)

    scanObj = os.scandir(path)
    returnCTime = None

    fileList = []
    for file in scanObj:
        if str(Name).lower() in file.name.lower():
            fileList.append(file.name)

    scanObj = os.scandir(path)

    if len(fileList) > 1:
        if loadingAnimator is not None:
            try:
                loadingAnimator.stop(method="None")
            except:
                pass
        if not skipFlag:
            print(f"\rThe Program found {Fore.CYAN + str(len(fileList)) + Style.RESET_ALL} files including the string {Fore.CYAN + str(Name)}.")
            inputValue = input("Do you want to select the [L]atest file or pick a file [M]anually?: ").lower()
        else:
            inputValue = "l"
        if inputValue == "m":
            returnValue = fileLister(path, Name)
            try:
                loadingAnimator.start()
            except:
                pass
            return returnValue

        elif inputValue == "l":
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
            try:
                loadingAnimator.start()
            except:
                pass
            return str(returnCTime[0])

        else:
            print(Fore.RED + "Wrong input detected defaulting to latest file.")

            for file in scanObj:
                if str(Name).lower() in file.name.lower():
                    if returnCTime is None:
                        returnCTime = [file.name, float(file.stat().st_ctime)]
                    elif returnCTime[1] < float(file.stat().st_ctime):
                        returnCTime = [file.name, float(file.stat().st_ctime)]
            return str(returnCTime[0])
    else:
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


def fileLister(pathInp, Name):
    path = Path(str(pathInp))
    scanObj = sorted(os.scandir(path), key=lambda t: t.stat().st_mtime, reverse=True)
    returnCTime = None
    referenceDict = {}
    counter = 1
    inputFlag = False

    for file in scanObj:
        if str(Name).lower() in file.name.lower():
            filename = path.joinpath(file.name)
            referenceDict[counter] = [file.name, filename.stat().st_size / 1024 / 1024, filename.stat().st_ctime]
            counter += 1

    while not inputFlag:
        for key, value in referenceDict.items():
            print(
                f"[{key}]: {Fore.CYAN + str(value[0]) + Style.RESET_ALL} | Size = {Fore.CYAN + str(round(value[1], 2))} mb {Style.RESET_ALL}| Creation Time = {Fore.CYAN + str(datetime.fromtimestamp(value[2]).strftime('%m-%d-%Y %H:%M:%S'))}")
        inputValue = input("Please pick a file by number: ")
        try:
            if int(inputValue) in list(referenceDict.keys()):
                return referenceDict[int(inputValue)][0]
            else:
                print("Invalid Input please try again")
        except:
            print("Invalid Input please try again")
