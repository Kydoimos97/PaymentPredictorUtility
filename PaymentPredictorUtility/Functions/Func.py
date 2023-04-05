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
from pathlib import Path
from colorama import Style, Fore, init
import pandas as pd
from datetime import datetime

init(autoreset=True)


def dateConvert(dataframe):
    """
The dateConvert function takes a dataframe as an argument and iterates through the columns of that dataframe.
If the column name contains &quot;date&quot; or &quot;next&quot;, it converts that column to datetime format. If not, it passes over
that column.

Args:
    dataframe: Pass the dataframe into the function

Returns:
    A dataframe with all date columns converted to datetime objects

Doc Author:
    Willem van der Schans, Trelent AI
"""
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
    """
The directoryScanner function is used to scan a directory for files that contain the string provided by the user.
The function will return either a list of all files containing the string or it will return one file based on what
the user wants. The function can also be set to only look in specific folders within a directory.

Args:
    Name: Search for a file in the directory
    SourcePath: Define the path to search in
    Folder: Specify a subfolder in the source path
    returnMethod: Determine if the function should return the first or last file found with
    loadingAnimator: Stop the loading animation while the user is selecting a file
    skipFlag: Skip the user input in case of multiple files found

Returns:
    A list of files that match the search string

Doc Author:
    Willem van der Schans, Trelent AI
"""
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
    """
The fileLister function takes in a path and a name of the file you are looking for.
It then scans the directory and returns all files that match your inputted name.
The function will then print out all files that match your search criteria, along with their size and creation time.
You can then select which file you want to use by typing in its number.

Args:
    pathInp: Specify the path to search for files in
    Name: Search for a file in the directory

Returns:
    The name of the file that was chosen

Doc Author:
    Willem van der Schans, Trelent AI
"""
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
