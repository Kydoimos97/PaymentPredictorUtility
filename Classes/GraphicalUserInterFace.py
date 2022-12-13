import datetime
import math
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")
from itertools import cycle
from pathlib import Path
from shutil import get_terminal_size
from threading import Thread
from time import sleep

import pandas as pd
from colorama import Fore, init, Style

from PaymentProb.Classes.CustomerClass import Customer
from PaymentProb.Classes.DataPrep import DataCleaner, dataMLPrep
from PaymentProb.Classes.ModelCreation import dataScaler, machineLearner
from PaymentProb.Classes.dataFrameLoader import dataFrameLoader
from PaymentProb.Classes.modelLoader import modelLoader
from PaymentProb.Functions.ErrorCodes import ErrorProcessor
from PaymentProb.Functions.Func import directoryScanner

init(autoreset=True, convert=True)
#init(autoreset=True)

# Design Methodology: Correct = Green, Error = Red, Input = White, Information/Processing = Blue, Warning/Action = Yellow,
# TODO Error Messages, Include SQL?, Set-up Automatic Test and file naming

class GraphicalUserInterface:

    def __init__(self):

        # Declare Variables
        self.verboseFlagBool = None
        self.skipFlag = None
        self.DebugFlag = False
        self.loopFlag = None
        self.loadingAnimator = None
        self.passFlag = None
        self.initFlag = True
        self.startTime = time.time()
        self.SourcePath = Path(os.getcwd())

        self.dataExists = True
        self.scalerObj = None
        self.modelObj = None
        self.sourceObj = None
        self.targetObj = None
        self.dummyObj = None

        self.exitFlag = False
        self.printString = ""

    def showGui(self):
        # Load all files and data
        print(self.asciiArt(), flush=True)
        # self.loadingAnimator = loadingAnimator("", "",
        #                                        "Data Loading Failed").start()
        #
        # self.loadingAnimator.stop()
        self.dataModelLoader()

        # Clear Console
        time.sleep(1)
        if not self.DebugFlag:
            os.system('cls||clear')

        # Get to main functionality
        while not self.exitFlag:
            if not self.DebugFlag:
                os.system('cls||clear')
            print(self.asciiArt(), flush=True)
            self.welcomeMethod()

        print(Fore.RED + "Exiting program..." + Style.RESET_ALL)
        time.sleep(1)

    def individualMethod(self):
        passFlag = False
        uniqueList = list(map(int, self.sourceObj.getDf()['acctrefno'].unique().tolist()))
        acctrefno = 0

        while not passFlag:
            print("Selection of random account Numbers:" + Fore.BLUE + f" {random.sample(uniqueList, 5)}" + Style.RESET_ALL)
            acctrefno = int(input("Please input an account Number: "))
            if acctrefno in uniqueList:
                passFlag = True
            else:
                print(
                    Fore.YELLOW + "Acctrefno not found in database, returning to Acctrefno input prompt." + Style.RESET_ALL)
                continue

        self.loadingAnimator = loadingAnimator(f"Fitting Model", "Model Fitting Completed",
                                               f"Model Fitting Failed").start()

        uniqueList = [acctrefno]
        StartingTime = time.time()
        BatchTime = time.time()
        customerDict = {}
        ErrorDict = {}
        loopLength = len(uniqueList)
        reportInterval = 1

        for x in range(loopLength):

            try:
                customerObj = Customer(uniqueList[x], self.sourceObj.getDf(), self.targetObj.getDf(),
                                       self.dummyObj.getDf(),
                                       self.modelObj.getModel(), self.scalerObj.getModel(), "Data")

                customerDict[uniqueList[x]] = [customerObj.accuracy, customerObj.certainty,
                                               customerObj.paymentPrediction.values,
                                               customerObj.nextPaymentProbability.values,
                                               customerObj.paymentCodes.values]  # Add prediction
            except Exception as e:
                ErrorDict[uniqueList[x]] = e
                pass

            if x == 0 or x == loopLength or x % reportInterval == 0:
                if self.verboseFlagBool:
                    print(Fore.GREEN +
                          f"\ncompleted {x + 1}/{(loopLength + 1)} at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
                          f"\ntotal runtime = {str(datetime.timedelta(seconds=math.ceil(time.time() - StartingTime)))} | "
                          f"\nBatch Run time = {str(datetime.timedelta(seconds=math.ceil(time.time() - BatchTime)))} | "
                          f"\nErrors Encountered = {len(ErrorDict)} | "
                          f"\nExpected time needed {str(datetime.timedelta(seconds=math.ceil((time.time() - StartingTime) / ((x + 1) / (loopLength + 1)))))}" + Style.RESET_ALL)
                    BatchTime = time.time()

        # Save Output
        if not os.path.exists(Path(self.SourcePath).joinpath('Output').joinpath(f"Individual{datetime.datetime.today().strftime('%Y%m%d')}")):
            os.mkdir(Path(self.SourcePath).joinpath('Output').joinpath(f"Individual{datetime.datetime.today().strftime('%Y%m%d')}"))

        df = pd.DataFrame.from_dict(customerDict, orient="index",
                                    columns=["Accuracy", "Certainty", "Prediction", "Payment Probability",
                                             "Payment Codes"])
        df.to_csv(self.SourcePath.joinpath('Output').joinpath(f"Individual{datetime.datetime.today().strftime('%Y%m%d')}").joinpath(f"{acctrefno}TestDataFrame{datetime.datetime.today().strftime('%Y%m%d_%H%M')}.csv"))

        dfError = pd.DataFrame.from_dict(ErrorDict, orient="index", columns=["Error"])
        dfError.to_csv(self.SourcePath.joinpath('Output').joinpath(f"Individual{datetime.datetime.today().strftime('%Y%m%d')}").joinpath(f"{acctrefno}ErrorDataFrame{datetime.datetime.today().strftime('%Y%m%d_%H%M')}.csv"))

        self.loadingAnimator.stop()

        print(Fore.GREEN + f"Output saved in Output/Individual{datetime.datetime.today().strftime('%Y%m%d')} folder." + Style.RESET_ALL, flush=True)
        if input("Do you want to exit the program? Y/N: ").lower() == "y":
            self.exitFlag = True
        else:
            pass

    def fullPredictionMethod(self):
        self.loadingAnimator = loadingAnimator("", "",
                                               "").start()

        uniqueList = list(self.sourceObj.getDf()["acctrefno"].unique())
        StartingTime = time.time()
        BatchTime = time.time()
        customerDict = {}
        ErrorDict = {}
        loopLength = len(uniqueList)
        reportInterval = 5000

        for x in range(loopLength):

            try:
                customerObj = Customer(uniqueList[x], self.sourceObj.getDf(), self.targetObj.getDf(),
                                       self.dummyObj.getDf(),
                                       self.modelObj.getModel(), self.scalerObj.getModel(), "Data")

                customerDict[uniqueList[x]] = [customerObj.accuracy, customerObj.certainty,
                                               customerObj.paymentPrediction.values,
                                               customerObj.nextPaymentProbability.values,
                                               customerObj.paymentCodes.values]  # Add prediction
            except Exception as e:
                ErrorDict[uniqueList[x]] = e
                pass

            if x == 0 or x == loopLength or x % reportInterval == 0:
                if self.verboseFlagBool:
                    print("\n" +
                          f"completed {x + 1}/{(loopLength + 1)} at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | \n"
                          f"total runtime = {str(datetime.timedelta(seconds=math.ceil(time.time() - StartingTime)))} | \n"
                          f"Batch Run time = {str(datetime.timedelta(seconds=math.ceil(time.time() - BatchTime)))} | \n"
                          f"Errors Encountered = {len(ErrorDict)} | \n"
                          f"Expected time needed {str(datetime.timedelta(seconds=math.ceil((time.time() - StartingTime) / ((x + 1) / (loopLength + 1)))))}")
                    BatchTime = time.time()

        # Save Output
        if not os.path.exists(Path(self.SourcePath).joinpath('Output').joinpath(f"Full{datetime.datetime.today().strftime('%Y%m%d')}")):
            os.mkdir(Path(self.SourcePath).joinpath('Output').joinpath(f"Full{datetime.datetime.today().strftime('%Y%m%d')}"))

        df = pd.DataFrame.from_dict(customerDict, orient="index",
                                    columns=["Accuracy", "Certainty", "Prediction", "Payment Probability",
                                             "Payment Codes"])
        df.to_csv(self.SourcePath.joinpath('Output').joinpath(f"Full{datetime.datetime.today().strftime('%Y%m%d')}").joinpath(f"CustomerTestDataFrame{datetime.datetime.today().strftime('%Y%m%d_%H%M')}.csv"))

        dfError = pd.DataFrame.from_dict(ErrorDict, orient="index", columns=["Error"])
        dfError.to_csv(self.SourcePath.joinpath('Output').joinpath(f"Full{datetime.datetime.today().strftime('%Y%m%d')}").joinpath(f"CustomerErrorDataFrame{datetime.datetime.today().strftime('%Y%m%d_%H%M')}.csv"))

        self.loadingAnimator.stop()

        print(Fore.GREEN + f"Output saved in Output/Full{datetime.datetime.today().strftime('%Y%m%d')} folder." + Style.RESET_ALL)
        if input("Do you want to exit the program? Y/N: ").lower() == "y":
            self.exitFlag = True
        else:
            pass

    def trainMethod(self):
        print(self.dividerSmall("Selection Menu", method="return", padding=8) + """
    1.""" + Fore.BLUE + """[O]""" + Style.RESET_ALL + """ptimal (Est. 4 Hours)
    2.""" + Fore.BLUE + """[F]""" + Style.RESET_ALL + """ast (Est. 5 Min)
    3.""" + Fore.BLUE + """[C]""" + Style.RESET_ALL + """ustom
    4.""" + Fore.BLUE + """[E]""" + Style.RESET_ALL + f"""xit{self.dividerSmall(None, method="return", fullLength=30)}
        """, flush=True)

        selVal = input("Please make a selection: ")

        if selVal.lower() == "o":
            print(
                Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.BLUE + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}" + Style.RESET_ALL)
            input("Press [enter] to continue or Ctrl+C to cancel...")

            self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                   "Creating ML Model Complete",
                                                   "Creating ML Model Failed").start()
            fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
            self.scalerObj = dataScaler(fileName,
                                        f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                        self.SourcePath, "Data")
            fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
            machineLearner(self.scalerObj.scaledDf, fileName,
                           f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                           ParamGrid="optimal",
                           Folder="Data")
            self.loadingAnimator.stop()
            input("Press [enter] to continue...")
        elif selVal.lower() == "f":
            print(
                Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.BLUE + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}" + Style.RESET_ALL)
            input("Press [enter] to continue or Ctrl+C to cancel...")

            self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                   "Creating ML Model Complete",
                                                   "Creating ML Model Failed").start()
            fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
            self.scalerObj = dataScaler(fileName,
                                        f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                        self.SourcePath, "Data")
            fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
            machineLearner(self.scalerObj.scaledDf, fileName,
                           f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                           ParamGrid="fast",
                           Folder="Data")
            self.loadingAnimator.stop()
            input("Press [enter] to continue...")
        elif selVal.lower() == "c":
            print(
                Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.BLUE + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}" + Style.RESET_ALL)
            n_estimatorsInp = input("Please input an integer to denote the N estimators: ")
            max_depthInp = input("Please input an integer to denote the max tree depth: ")

            ParamGrid = {"n_estimators": [n_estimatorsInp],
                         "max_depth": [max_depthInp],
                         "alpha": [1],
                         "learning_rate": [0.1],
                         "colsample_bytree": [0.8],
                         "lambda": [1],
                         "subsample": [1]}

            input(
                f"Press [enter] to continue learning with" + Fore.BLUE + f"N_est = {n_estimatorsInp}" + Style.RESET_ALL + " and " +
                Fore.BLUE + f"Max depth = {max_depthInp}" + Style.RESET_ALL + " or Ctrl+C to cancel...")

            self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                   "Creating ML Model Complete",
                                                   "Creating ML Model Failed").start()
            fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
            self.scalerObj = dataScaler(fileName,
                                        f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                        self.SourcePath, "Data")
            fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
            machineLearner(self.scalerObj.scaledDf, fileName,
                           f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                           ParamGrid=ParamGrid,
                           Folder="Data")
            self.loadingAnimator.stop()
            input("Press [enter] to continue...")
        elif selVal.lower() == "e":
            if input(Fore.RED + "Are you sure you want to exit? Y/N:" + Style.RESET_ALL).lower() == "y":
                self.exitFlag = True
            else:
                pass

    def methodSelector(self, selection):

        if selection == "i":
            self.passFlag = True
            print(Fore.BLUE + """Individual Method Selected
            """ + Style.RESET_ALL, flush=True)
            self.individualMethod()

        elif selection == "f":
            print(Fore.BLUE + "Full Prediction Method Selected" + Fore.YELLOW +
                  "\nWARNING: This method can take a lot of time (Est. 5 hours)." + Style.RESET_ALL, flush=True)

            selVal = input("Are you sure you want to continue? Y/N:")

            if selVal.lower() == "y":
                self.fullPredictionMethod()
            elif selVal.lower() == "n":
                print(Fore.RED + """Method cancelled returning to main menu""" + Style.RESET_ALL, flush=True)

        elif selection == "t":
            self.passFlag = True
            print(Fore.BLUE + "Model Training Method Selected" + Fore.YELLOW + "\nWARNING: This method can take a lot of time when using Optimal Parameters (Est. 4.5 hours)." + Style.RESET_ALL, flush=True)

            selVal = input("Are you sure you want to continue? Y/N:")

            if selVal.lower() == "y":
                self.trainMethod()
            elif selVal.lower() == "n":
                print(Fore.RED + "Method cancelled returning to main menu" + Style.RESET_ALL, flush=True)

        elif selection == "e":
            if input(Fore.RED + "Are you sure you want to exit? Y/N:" + Style.RESET_ALL).lower() == "y":
                self.exitFlag = True
            else:
                pass

        else:
            ErrorProcessor(901, "__main__", "unknown")

    def welcomeMethod(self):
        self.passFlag = False

        print(self.dividerSmall("Selection Menu", method="return", padding=8) + """
    1.""" + Fore.BLUE + """[I]""" + Style.RESET_ALL + """ndividual prediction
    2.""" + Fore.BLUE + """[F]""" + Style.RESET_ALL + """ull set Prediction
    3.""" + Fore.BLUE + """[T]""" + Style.RESET_ALL + """rain Model
    4.""" + Fore.BLUE + """[E]""" + Style.RESET_ALL + f"""xit{self.dividerSmall(None, method="return", fullLength=30)}
        """, flush=True)

        while not self.passFlag:
            try:
                selVal = input("Please select a method: ").lower()
                self.methodSelector(selVal)
                self.passFlag = True
            except ValueError as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}")  # Debug
                print(Fore.RED + "Please make a valid selection [I,F,T,E]" + Style.RESET_ALL)
                continue

    def dataModelLoader(self):

        # Reload Overwrite
        retrainFlag = False
        reloadFlag = False
        self.DebugFlag = False
        self.verboseFlagBool = False

        self.dividerSmall(title="Initialization")
        skipInput = input("Do you want to skip manual inputs?" + Style.RESET_ALL + " Y/N or [D]ebug: ").lower()

        if skipInput == "y":
            self.skipFlag = True
        elif skipInput == "d" or skipInput == "debug":
            print(Fore.RED + "Debug Mode Activated!!!" + Style.RESET_ALL)
            self.skipFlag = False
            self.DebugFlag = True
            self.verboseFlagBool = True

        else:
            self.skipFlag = False

        while self.initFlag:

            if self.initFlag and not reloadFlag:
                recreateInp = input(
                    "Did" + Fore.BLUE + " tblXmain_transactions.csv" + Style.RESET_ALL + " get Updated? Y/N:" + Fore.RED + " [Y will recreate all files]: " + Style.RESET_ALL)
                if recreateInp.lower() == "y":
                    reloadFlag = True
                    print(
                        Fore.YELLOW + " Note: The next question does not impact results by a large margin. \n    Only do this if you can spare the time or if the source table variables (additions or deletions) changed." + Style.RESET_ALL)
                    if input(
                            "Do you want to retrain the scaler and ensemble tree model? " + Fore.BLUE + "(Est. Time 4 Hours)" + Style.RESET_ALL + " Y/N:").lower() == "y":
                        retrainFlag = True
                else:
                    pass
            else:
                reloadFlag = False
                pass

            self.dividerSmall(title="Data Folder")

            # Data Folder
            if not os.path.exists(self.SourcePath.joinpath("Data")):
                if input(
                        Fore.RED + "[Data] folder not found. \n" + Style.RESET_ALL + "Do you want to create a folder? Y/N:").lower() == "y":
                    os.mkdir((self.SourcePath.joinpath("Data")))
                    print(
                        Fore.GREEN + f"\nFolder [Data] created at: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                    input(
                        Fore.YELLOW + "Please move any files to the created [data] folder." + Style.RESET_ALL + " Press [enter] when ready to continue...")
                else:
                    print(
                        Fore.RED + f"\nAction cancelled please make sure this folder exists:\n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                    print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                    time.sleep(3)
                    self.exitFlag = True
                    return
            else:
                print(Fore.GREEN + "Data Folder Found." + Style.RESET_ALL)

            self.dividerSmall(title="Output Folder")

            # OUTPUT FOLDER
            if not os.path.exists(self.SourcePath.joinpath("Output")):
                if input(
                        Fore.RED + "[Output] folder not found. \n" + Style.RESET_ALL + "Do you want to create a folder? Y/N: ").lower() == "y":
                    os.mkdir((self.SourcePath.joinpath("Output")))
                    print(
                        Fore.GREEN + f"Folder [Output] created at: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Output')}" + Style.RESET_ALL)
                else:
                    print(
                        Fore.RED + f"\nAction cancelled please make sure this folder exists: \n" + Fore.BLUE + f"{self.SourcePath.joinpath('Output')}" + Style.RESET_ALL)
                    print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                    time.sleep(3)
                    self.exitFlag = True
                    return
            else:
                print(Fore.GREEN + "Output Folder Found." + Style.RESET_ALL)

            self.dividerSmall(title="dfClean Data")

            # DFCLEAN
            try:
                if reloadFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading dfClean...", "Loading dfClean Complete",
                                                       "Loading dfClean Failed",
                                                       0.05).start()

                fileName = directoryScanner("dfClean", self.SourcePath, Folder="Data", returnMethod="Last")
                self.sourceObj = dataFrameLoader(fileName, self.SourcePath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}")  # Debug
                if not reloadFlag:
                    self.loadingAnimator.stop(method="error")
                    if input(
                            Fore.RED + "[dfClean] not found!!! " + Style.RESET_ALL + "Do you want to create this file? Y/N:").lower() == "y":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.BLUE + "[tblXmain_transactions.csv]" + Fore.YELLOW + " exists the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue" + " (Est. Time = 2 Minutes)..." + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Cleaning Data Set...", "Cleaning Complete",
                                                               "Cleaning Failed").start()
                        DataCleaner("tblXmain_transactions.csv",
                                    f"dfClean{datetime.datetime.today().strftime('%Y_%m_%d')}.csv", "Data",
                                    path=self.SourcePath, verbose=self.verboseFlagBool)
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\nAction cancelled please make " + Fore.BLUE + "[dfClean.csv]" + Fore.RED + " exists in \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.GREEN + "Rebuilding CleanDF" + Fore.YELLOW + f"Make sure " + Fore.BLUE + "[tblXmain_transactions.csv]" + Fore.YELLOW + " exists the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                    if not self.skipFlag:
                        input("Press [enter] to continue" + " (Est. Time = 2 Minutes)..." + Style.RESET_ALL)
                    self.loadingAnimator = loadingAnimator("Cleaning Data Set...", "Cleaning Complete",
                                                           "Cleaning Failed").start()
                    fileName = directoryScanner("tblXmain_transactions", self.SourcePath, Folder="Data",
                                                returnMethod="Last")
                    DataCleaner(fileName, f"dfClean{datetime.datetime.today().strftime('%Y_%m_%d')}.csv", "Data",
                                path=self.SourcePath, verbose=self.verboseFlagBool)
                    self.loadingAnimator.stop()

            self.dividerSmall(title="dfDummy & dfTarget Data")

            # DFDUMMY AND DFTARGET (X AND Y)
            try:
                if reloadFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading dfDummy and dfTarget...",
                                                       "Loading dfDummy and dfTarget Complete",
                                                       "Loading dfDummy and dfTarget Failed").start()
                fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                self.dummyObj = dataFrameLoader(fileName, self.SourcePath, "Data", verbose=self.verboseFlagBool)
                fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                self.targetObj = dataFrameLoader(fileName, self.SourcePath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}")  # Debug
                if not reloadFlag:
                    self.loadingAnimator.stop(method="error")
                    if input(
                            Fore.RED + "[dfDummy] or [dfTarget] not found!!! " + Style.RESET_ALL + f"Do you want to create this file? Y/N:").lower() == "y":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfClean.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue " + " (Est. Time = 1 Minute)..." + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Machine Learning Preparation...",
                                                               "Machine Learning Preparation Complete",
                                                               "Machine Learning Preparation Failed").start()
                        fileName = directoryScanner("dfClean", self.SourcePath, Folder="Data", returnMethod="Last")
                        dataMLPrep(fileName, f"dfDummy{datetime.datetime.today().strftime('%Y_%m_%d')}.csv",
                                   f"dfTarget{datetime.datetime.today().strftime('%Y_%m_%d')}.csv", "Data",
                                   verbose=self.verboseFlagBool)
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Action cancelled please make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.RED + " and " + Fore.BLUE + "[dfTarget.csv]" + Fore.RED + " exist in \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.GREEN + "Rebuilding dfDummy and dfTarget. " + Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfClean.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                    if not self.skipFlag:
                        input("Press [enter] to continue " + " (Est. Time = 1 Minute)..." + Style.RESET_ALL)
                    self.loadingAnimator = loadingAnimator("Machine Learning Preparation...",
                                                           "Machine Learning Preparation Complete",
                                                           "Machine Learning Preparation Failed").start()
                    fileName = directoryScanner("dfClean", self.SourcePath, Folder="Data", returnMethod="Last")
                    dataMLPrep(fileName, f"dfDummy{datetime.datetime.today().strftime('%Y_%m_%d')}.csv",
                               f"dfTarget{datetime.datetime.today().strftime('%Y_%m_%d')}.csv", "Data", verbose=self.verboseFlagBool)
                    self.loadingAnimator.stop()

            self.dividerSmall(title="Scaling Model")

            # SCALING MODEL
            try:
                if reloadFlag and retrainFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading Scaling Model...", "Loading Scaling Model Complete",
                                                       "Loading Scaling Model Failed").start()
                fileName = directoryScanner("scaler", self.SourcePath, Folder="Data", returnMethod="Last")
                self.scalerObj = modelLoader(fileName, self.SourcePath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}")  # Debug
                if not reloadFlag and not retrainFlag:
                    self.loadingAnimator.stop(method="error")
                    if input(
                            Fore.RED + "[scaler.sav] not found or corrupted!!! " + Style.RESET_ALL + f"Do you want to create this file? Y/N:").lower() == "y":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue " + " (Est. Time = 30 Seconds)..." + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating Scaling Model...",
                                                               "Creating Scaling Model Complete",
                                                               "Creating Scaling Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Action cancelled please make " + Fore.BLUE + "[scaler.sav]" + Fore.RED + " exists in \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.BLUE + "Scaler Model will be retrained in the next section.")
                    pass

            self.dividerSmall(title="Machine Learning Model")

            # ML MODEL
            try:
                if reloadFlag and retrainFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading ML Model...", "Loading ML Model Complete",
                                                       "Loading ML Model Failed").start()
                fileName = directoryScanner("model", self.SourcePath, Folder="Data", returnMethod="Last")
                self.modelObj = modelLoader(fileName, self.SourcePath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}")  # Debug
                if not reloadFlag and not retrainFlag:
                    self.loadingAnimator.stop(method="error")
                    methodChoice = input(
                        Fore.RED + "[model.sav] not found or corrupted!!! " + Style.RESET_ALL + "Do you want to create this file using " + Fore.BLUE + "[O]" + Style.RESET_ALL + "ptimal, (Est. 4 Hours)" + Fore.BLUE + "[F]" + Style.RESET_ALL + "ast (Est. 5 min), or " + Fore.BLUE + "[C]" + Style.RESET_ALL + "ustom parameters? or " + Fore.BLUE + "[C]" + Style.RESET_ALL + "ancel this action?:")
                    if methodChoice.lower() == "o":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.BLUE + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue or Ctrl+C to cancel " + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                                       ParamGrid="optimal",
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    elif methodChoice.lower() == "f":
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue or Ctrl+C to cancel " + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                                       ParamGrid="fast",
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    elif methodChoice.lower() == "c":
                        n_estimatorsInp = input("\nPlease input an integer to denote the N estimators: ")
                        max_depthInp = input("\nPlease input an integer to denote the max tree depth: ")

                        ParamGrid = {"n_estimators": [n_estimatorsInp],
                                     "max_depth": [max_depthInp],
                                     "alpha": [1],
                                     "learning_rate": [0.1],
                                     "colsample_bytree": [0.8],
                                     "lambda": [1],
                                     "subsample": [1]}

                        if not self.skipFlag:
                            input(
                                f"Press [enter] to continue learning with" + Fore.BLUE + f"N_est = {n_estimatorsInp}" + Style.RESET_ALL + " and " +
                                Fore.BLUE + f"Max depth = {max_depthInp}" + Style.RESET_ALL + " or Ctrl+C to cancel...")

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                                       ParamGrid=ParamGrid,
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Action cancelled please make " + Fore.BLUE + "[model.sav]" + Fore.RED + " exists in \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.GREEN + "Rebuilding Scaler and Ensemble Models. " + Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfClean.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                    methodChoice = input(
                        "Do you want to train the model with " + Fore.BLUE + "[O]" + Style.RESET_ALL + "ptimal, " + Fore.BLUE + "[F]" + Style.RESET_ALL + "ast, or " + Fore.BLUE + "[C]" + Style.RESET_ALL + "ustom parameters?")
                    if methodChoice.lower() == "o":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.BLUE + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.BLUE + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}" + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue or Ctrl+C to cancel " + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                                       ParamGrid="optimal",
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    elif methodChoice.lower() == "f":
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue or Ctrl+C to cancel " + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                                       ParamGrid="fast",
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    elif methodChoice.lower() == "c":
                        n_estimatorsInp = input("\nPlease input an integer to denote the N estimators: ")
                        max_depthInp = input("\nPlease input an integer to denote the max tree depth: ")

                        ParamGrid = {"n_estimators": [n_estimatorsInp],
                                     "max_depth": [max_depthInp],
                                     "alpha": [1],
                                     "learning_rate": [0.1],
                                     "colsample_bytree": [0.8],
                                     "lambda": [1],
                                     "subsample": [1]}

                        if not self.skipFlag:
                            input(
                                f"Press [enter] to continue learning with" + Fore.BLUE + f"N_est = {n_estimatorsInp}" + Style.RESET_ALL + " and " +
                                Fore.BLUE + f"Max depth = {max_depthInp}" + Style.RESET_ALL + " or Ctrl+C to cancel...")

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.SourcePath, Folder="Data", returnMethod="Last")
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%Y_%m_%d')}.sav",
                                                    self.SourcePath, "Data")
                        fileName = directoryScanner("dfTarget", self.SourcePath, Folder="Data", returnMethod="Last")
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%Y_%m_%d')}.sav", self.SourcePath,
                                       ParamGrid=ParamGrid,
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Wrong input detected" + Fore.BLUE + "[model.sav]" + Fore.RED + " exists in \n" + Fore.BLUE + f"  {self.SourcePath.joinpath('Data')}"  + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return



            if reloadFlag:
                self.initFlag = True
            else:
                self.initFlag = False

    @staticmethod
    def asciiArt():
        asciiArt = Fore.GREEN + """
        ____________________________________________________         
                               _____                                          
                              /#####\                      
                             /#######\                    
                            /#########\                   
                           /###########\                 
                          /#####/‾\#####\                
                         /#####/   \#####\              
                        /#####/     \#####\             
                       /#####/       \#####\         
                      /#####/         \#####\          
                     /#####/           \#####\        
                    /#####/_____________\#####\      
                   /###########################\\
                  /#############################\   
                 /#####/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\#####\\
                /#####/                     \#####\\
               /#####/                       \#####\                     
               ‾‾‾‾‾‾                         ‾‾‾‾‾‾           
                      Ⓒ Avid Acceptance LLC                 
              Payment Probability Prediction Utility        
        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾""" + Style.RESET_ALL

        return asciiArt

    @staticmethod
    def asciiArtSmall():
        asciiArt = Fore.GREEN + """
            /\    
           /  \   
          / /\ \  
         / ____ \ 
        /_/    \_\
           Avid
        """ + Style.RESET_ALL

        return asciiArt

    @staticmethod
    def dividerSmall(title=None, method="print", padding=None, fullLength=None):

        titleLength = len(str(title))

        if fullLength is None:
            fullLength = 46
        else:
            fullLength = fullLength

        halfLength = int(fullLength / 2)

        if padding is not None:
            leftLength = padding
            rightLength = padding
        else:
            leftLength = (halfLength - math.ceil(titleLength / 2))
            rightLength = halfLength - math.floor(titleLength / 2)

        if title is None:
            dividerText = f"""
{"-" * fullLength}"""
        else:
            dividerText = f"""
{"-" * leftLength}{str(title).lower().capitalize()}{"-" * rightLength}"""

        if method.lower() == "print":
            print(Fore.CYAN + dividerText + Style.RESET_ALL)
        elif method.lower() == "return":
            return Fore.CYAN + dividerText + Style.RESET_ALL


class loadingAnimator:
    def __init__(self, desc="Loading...", end="Done!", error="Loading Failed", timeout=0.3):
        self.desc = Fore.BLUE + desc + Style.RESET_ALL
        self.end = Fore.GREEN + end + Style.RESET_ALL
        self.timeout = timeout
        self.error = Fore.RED + error + Style.RESET_ALL
        self.startTime = time.time()

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = [Fore.BLUE + f"⢿" + Style.RESET_ALL,
                      Fore.BLUE + f"⣻" + Style.RESET_ALL,
                      Fore.BLUE + f"⣽" + Style.RESET_ALL,
                      Fore.BLUE + f"⣾" + Style.RESET_ALL,
                      Fore.BLUE + f"⣷" + Style.RESET_ALL,
                      Fore.BLUE + f"⣯" + Style.RESET_ALL,
                      Fore.BLUE + f"⣟" + Style.RESET_ALL,
                      Fore.BLUE + f"⡿" + Style.RESET_ALL]
        self.done = False

    def start(self):
        self._thread.start()
        self.startTime = time.time()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c} T+ {datetime.timedelta(seconds=round(time.time() - self.startTime, 0))}",
                  flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self, method=""):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        if method.lower() == "error":
            print(f"\r{self.error}", flush=True)
        if method.lower() == "none":
            pass
        else:
            print(f"\r{self.end}", flush=True)
        time.sleep(0.5)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
