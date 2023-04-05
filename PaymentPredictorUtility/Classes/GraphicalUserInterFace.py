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

import datetime
import math
import os
import random
import time
import warnings

import pyodbc

warnings.filterwarnings("ignore")
from itertools import cycle
from pathlib import Path
from shutil import get_terminal_size
from threading import Thread
from time import sleep

import pandas as pd
from colorama import Fore, init, Style

from PaymentPredictorUtility.Classes.CustomerClass import Customer
from PaymentPredictorUtility.Classes.DataPrep import DataCleaner, dataMLPrep
from PaymentPredictorUtility.Classes.ModelCreation import dataScaler, machineLearner
from PaymentPredictorUtility.Classes.dataFrameLoader import dataFrameLoader
from PaymentPredictorUtility.Classes.modelLoader import modelLoader
from PaymentPredictorUtility.Functions.ErrorCodes import ErrorProcessor
from PaymentPredictorUtility.Functions.Func import directoryScanner

init(autoreset=True, convert=True)




class GraphicalUserInterface:

    def __init__(self):

       
        """
    The __init__ function is called when the class is instantiated.
    It sets up all of the variables that will be used throughout the program.


    Args:
        self: Represent the instance of the class

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.verboseFlagBool = None
        self.skipFlag = None
        self.DebugFlag = False
        self.loopFlag = None
        self.loadingAnimator = None
        self.passFlag = None
        self.initFlag = True
        self.startTime = time.time()
        self.SourcePath = Path(os.getcwd())
        self.docPath = Path(os.path.expanduser('~/Documents')).joinpath("AvidPaymentPredictor")

        self.dataExists = True
        self.scalerObj = None
        self.modelObj = None
        self.sourceObj = None
        self.targetObj = None
        self.dummyObj = None

        self.exitFlag = False
        self.printString = ""

    def showGui(self):
       
        """
    The showGui function is the main function of the program. It loads all files and data, clears the console,
    and then gets to main functionality. The while loop will continue until exitFlag is set to True.

    Args:
        self: Refer to the current object

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        print(self.asciiArt(), flush=True)
        self.dataModelLoader()

       
        time.sleep(1)
        if not self.DebugFlag:
            os.system('cls||clear')

       
        while not self.exitFlag:
            if not self.DebugFlag:
                os.system('cls||clear')
            print(self.asciiArt(), flush=True)
            self.welcomeMethod()

        print(Fore.RED + "Exiting program..." + Style.RESET_ALL)
        time.sleep(1)

    def individualMethod(self):
        """
    The individualMethod function is used to predict the next payment of a single customer.
        The user will be prompted to input an account number, and the program will return a prediction for that specific customer.
        This function is useful when you want to test out how accurate your model is on individual customers.

    Args:
        self: Represent the instance of the class

    Returns:
        A dictionary of the account number, accuracy, certainty and prediction

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        passFlag = False
        uniqueList = list(map(int, self.sourceObj.getDf()['acctrefno'].unique().tolist()))
        acctrefno = 0

        while not passFlag:
            print(
                "Selection of random account Numbers:" + Fore.CYAN + f" {random.sample(uniqueList, 5)}" + Style.RESET_ALL)
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
                                               customerObj.paymentCodes.values] 
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

       
        if not os.path.exists(Path(self.docPath).joinpath('Output').joinpath(
                f"Individual{datetime.datetime.today().strftime('%m%d%Y')}")):
            os.mkdir(Path(self.docPath).joinpath('Output').joinpath(
                f"Individual{datetime.datetime.today().strftime('%m%d%Y')}"))

        df = pd.DataFrame.from_dict(customerDict, orient="index",
                                    columns=["Accuracy", "Certainty", "Prediction", "Payment Probability",
                                             "Payment Codes"])
        df.to_csv(self.docPath.joinpath('Output').joinpath(
            f"Individual{datetime.datetime.today().strftime('%m%d%Y')}").joinpath(
            f"{acctrefno}_TestDataFrame.csv"))

        dfError = pd.DataFrame.from_dict(ErrorDict, orient="index", columns=["Error"])
        dfError.to_csv(self.docPath.joinpath('Output').joinpath(
            f"Individual{datetime.datetime.today().strftime('%m%d%Y')}").joinpath(
            f"{acctrefno}_ErrorDataFrame.csv"))

        self.loadingAnimator.stop()

        print(
            f"Output = {Fore.CYAN + str(customerDict) + Style.RESET_ALL} \n Error = {Fore.CYAN + str(ErrorDict) + Style.RESET_ALL} \n")

        print(
            Fore.GREEN + f"Output saved in {str(self.docPath.joinpath('Output'))}\Individual{datetime.datetime.today().strftime('%m%d%Y')} folder." + Style.RESET_ALL,
            flush=True)
        if input("Do you want to exit the program? Y/N: ").lower() == "y":
            self.exitFlag = True
        else:
            pass

    def fullPredictionMethod(self):
        """
    The fullPredictionMethod function is the main function of the program. It takes in all of the data from
    the sourceObj, targetObj, dummyObj and modelObj objects and uses them to create a Customer object for each unique account number.
    The customer object then runs through its own prediction method which returns an accuracy score, certainty score (how confident it is that it's correct),
    a predicted payment amount (if any), a probability that there will be a payment made next month and what type of payment code was used to make this prediction.
    This information is then saved into two csv files: one with all successful predictions and

    Args:
        self: Represent the instance of the class

    Returns:
        A dataframe with the following columns:

    Doc Author:
        Willem van der Schans, Trelent AI
    """
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
                                               customerObj.paymentCodes.values] 
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

       
        if not os.path.exists(
                Path(self.docPath).joinpath('Output').joinpath(
                    f"Full{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}")):
            os.mkdir(
                Path(self.docPath).joinpath('Output').joinpath(
                    f"Full{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}"))

        df = pd.DataFrame.from_dict(customerDict, orient="index",
                                    columns=["Accuracy", "Certainty", "Prediction", "Payment Probability",
                                             "Payment Codes"])
        df.to_csv(
            self.docPath.joinpath('Output').joinpath(
                f"Full{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}").joinpath(
                f"CustomerTestDataFrame{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv"))

        dfError = pd.DataFrame.from_dict(ErrorDict, orient="index", columns=["Error"])
        dfError.to_csv(
            self.docPath.joinpath('Output').joinpath(
                f"Full{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}").joinpath(
                f"CustomerErrorDataFrame{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv"))

        self.loadingAnimator.stop()

        print(
            Fore.GREEN + f"Output saved in {str(self.docPath.joinpath('Output'))}/Full{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')} folder." + Style.RESET_ALL)
        if input("Do you want to exit the program? Y/N: ").lower() == "y":
            self.exitFlag = True
        else:
            pass

    def trainMethod(self):
        """
    The trainMethod function is the main function that allows the user to train a machine learning model.
    It has three options: optimal, fast, and custom. Optimal will take about 4 hours to run while fast will take about 5 minutes.
    Custom allows you to specify your own parameters for training.

    Args:
        self: Access the class attributes and methods

    Returns:
        A string

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        print(self.dividerSmall("Selection Menu", method="return", padding=8) + """
    1.""" + Fore.CYAN + """[O]""" + Style.RESET_ALL + """ptimal (Est. 4 Hours)
    2.""" + Fore.CYAN + """[F]""" + Style.RESET_ALL + """ast (Est. 5 Min)
    3.""" + Fore.CYAN + """[C]""" + Style.RESET_ALL + """ustom
    4.""" + Fore.CYAN + """[E]""" + Style.RESET_ALL + f"""xit{self.dividerSmall(None, method="return", fullLength=30)}
        """, flush=True)

        selVal = input("Please make a selection: ")

        if selVal.lower() == "o":
            print(
                Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.CYAN + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
            input("Press [enter] to continue or Ctrl+C to cancel...")

            self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                   "Creating ML Model Complete",
                                                   "Creating ML Model Failed").start()
            fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                        loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
            self.scalerObj = dataScaler(fileName,
                                        f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                        self.docPath, "Data")
            fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                        loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
            machineLearner(self.scalerObj.scaledDf, fileName,
                           f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
                           ParamGrid="optimal",
                           Folder="Data")
            self.loadingAnimator.stop()
            input("Press [enter] to continue...")
        elif selVal.lower() == "f":
            print(
                Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.CYAN + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
            input("Press [enter] to continue or Ctrl+C to cancel...")

            self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                   "Creating ML Model Complete",
                                                   "Creating ML Model Failed").start()
            fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                        loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
            self.scalerObj = dataScaler(fileName,
                                        f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                        self.docPath, "Data")
            fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                        loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
            machineLearner(self.scalerObj.scaledDf, fileName,
                           f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
                           ParamGrid="fast",
                           Folder="Data")
            self.loadingAnimator.stop()
            input("Press [enter] to continue...")
        elif selVal.lower() == "c":
            print(
                Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.CYAN + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
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
                f"Press [enter] to continue learning with" + Fore.CYAN + f"N_est = {n_estimatorsInp}" + Style.RESET_ALL + " and " +
                Fore.CYAN + f"Max depth = {max_depthInp}" + Style.RESET_ALL + " or Ctrl+C to cancel...")

            self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                   "Creating ML Model Complete",
                                                   "Creating ML Model Failed").start()
            fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                        loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
            self.scalerObj = dataScaler(fileName,
                                        f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                        self.docPath, "Data")
            fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                        loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
            machineLearner(self.scalerObj.scaledDf, fileName,
                           f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
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

        """
    The methodSelector function is the main menu for the program. It allows users to select which method they would like to use.
        The function takes in a single parameter, selection, which is used as a switch statement to determine what method should be called next.

    Args:
        self: Access the class variables and functions
        selection: Determine which method to call

    Returns:
        A boolean value of true or false

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if selection == "i":
            self.passFlag = True
            print(Fore.CYAN + """Individual Method Selected
            """ + Style.RESET_ALL, flush=True)
            self.individualMethod()

        elif selection == "f":
            print(Fore.CYAN + "Full Prediction Method Selected" + Fore.YELLOW +
                  "\nWARNING: This method can take a lot of time (Est. 5 hours)." + Style.RESET_ALL, flush=True)

            selVal = input("Are you sure you want to continue? Y/N:")

            if selVal.lower() == "y":
                self.fullPredictionMethod()
            elif selVal.lower() == "n":
                print(Fore.RED + """Method cancelled returning to main menu""" + Style.RESET_ALL, flush=True)

        elif selection == "t":
            self.passFlag = True
            print(
                Fore.CYAN + "Model Training Method Selected" + Style.RESET_ALL,
                flush=True)

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
        """
    The welcomeMethod function is the first function that is called when the program starts.
    It prints a menu of options to select from and then calls methodSelector with the user's selection.

    Args:
        self: Access the class variables and methods

    Returns:
        A value to the methodselector function

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.passFlag = False

        print(self.dividerSmall("Selection Menu", method="return", padding=8) + """
    1.""" + Fore.CYAN + """[I]""" + Style.RESET_ALL + """ndividual prediction
    2.""" + Fore.CYAN + """[F]""" + Style.RESET_ALL + """ull set Prediction
    3.""" + Fore.CYAN + """[T]""" + Style.RESET_ALL + """rain Model
    4.""" + Fore.CYAN + """[E]""" + Style.RESET_ALL + f"""xit{self.dividerSmall(None, method="return", fullLength=30)}
        """, flush=True)

        while not self.passFlag:
            try:
                selVal = input("Please select a method: ").lower()
                self.methodSelector(selVal)
                self.passFlag = True
            except ValueError as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}") 
                print(Fore.RED + "Please make a valid selection [I,F,T,E]" + Style.RESET_ALL)
                continue

    def dataModelLoader(self):
        """
    The dataModelLoader function is used to load all the necessary files for the AvidPaymentPredictor.
    This function will check if any of the files are missing and create them if needed.
    The dataModelLoader function also checks for updates in tblXmain_transactions and will recreate all
    necessary files if an update is detected.
    The dataModelLoader function is then used to load the scaler and ML model from the Data folder.
    If it does not exist, it will prompt user to create one using either optimal, fast or custom parameters.

    Args:
    self: Refer to the object itself

    Returns:
    The scalerobj and modelobj objects

    Doc Author:
    Willem van der Schans, Trelent AI
    """
       
        retrainFlag = False
        reloadFlag = False
        self.DebugFlag = False
        self.verboseFlagBool = False

        self.dividerSmall(title="Initialization")
        skipInput = input("Do you want to skip manual inputs?" + Style.RESET_ALL + " Y/N: ").lower()

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
                    "Did" + Fore.CYAN + " tblXmain_transactions.csv" + Style.RESET_ALL + " get Updated? Y/N:" + Fore.RED + " [Y will recreate all files]: " + Style.RESET_ALL)
                if recreateInp.lower() == "y":
                    reloadFlag = True
                    print(
                        Fore.YELLOW + " Note: The next question does not impact results by a large margin. \n    Only do this if you can spare the time or if the source table variables (additions or deletions) changed." + Style.RESET_ALL)
                    if input(
                            "Do you want to retrain the scaler and ensemble tree model? Y/N:").lower() == "y":
                        retrainFlag = True
                else:
                    pass
            else:
                reloadFlag = False
                pass

           
            if not os.path.exists(self.docPath):
                os.mkdir(self.docPath)

            self.dividerSmall(title="Data Folder")

           
            if not os.path.exists(self.docPath.joinpath("Data")):
                if input(
                        Fore.RED + "[Data] folder not found. \n" + Style.RESET_ALL + "Do you want to create a folder? Y/N:").lower() == "y":
                    os.mkdir((self.docPath.joinpath("Data")))
                    print(
                        Fore.GREEN + f"\nFolder [Data] created at: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                    input(
                        Fore.YELLOW + "Please move any files to the created [data] folder." + Style.RESET_ALL + " Press [enter] when ready to continue...")
                else:
                    print(
                        Fore.RED + f"\nAction cancelled please make sure this folder exists:\n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                    print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                    time.sleep(3)
                    self.exitFlag = True
                    return
            else:
                print(Fore.GREEN + "Data Folder Found." + Style.RESET_ALL)

            self.dividerSmall(title="tblXmain_transactions")

           
            try:
                fileName = directoryScanner("tblXmain_transactions", self.docPath, Folder="Data",
                                            returnMethod="Last", loadingAnimator=self.loadingAnimator,
                                            skipFlag=self.skipFlag)
                if fileName is None or fileName == "":
                    raise FileNotFoundError
                print(Fore.GREEN + "tblXmain_transactions Found." + Style.RESET_ALL)

            except Exception as e:
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}") 

                print(Fore.YELLOW + "Note getting tblXmain_transaction will only work on the Avid network!!!")
                if input(
                        Fore.RED + "[tblXmain_transactions] not found. \n" + Style.RESET_ALL + "Do you want to download tblXmain_transactions? Y/N:").lower() == "y":
                    try:
                        userName = input("Please enter SqlServer username:")
                        passWord = input("Please enter your SqlServer password:")
                        self.loadingAnimator = loadingAnimator("Getting tblXmain_transactions...",
                                                               "Getting tblXmain_transactions Complete",
                                                               "Getting tblXmain_transactions Failed").start()
                        server = '10.6.0.48'
                        database = 'warehouse'
                        username = userName
                        password = passWord

                        cnxn = pyodbc.connect(
                            'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';ENCRYPT=no;UID=' + username + ';PWD=' + password,
                            autocommit=True)

                        sql_query = pd.read_sql_query("SELECT * FROM tblXmain_transactions", cnxn)
                        df = pd.DataFrame(sql_query)
                        df.to_csv(Path(os.path.expanduser('~/Documents')).joinpath("AvidPaymentPredictor").joinpath(
                            "Data").joinpath(
                            f"tblXmain_transactions{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv"))
                        cnxn.close()
                        self.loadingAnimator.stop()
                    except Exception as e:
                        try:
                            self.loadingAnimator.stop(method="error")
                        except:
                            pass
                        if self.DebugFlag:
                            print(Fore.RED + f"DEBUG MESSAGE::: {e}") 
                        print(
                            Fore.RED + f"\n Getting table failed " + Fore.CYAN + "[tblXmain_transaction.csv]" + Fore.RED + " Exists in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}")
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.RED + f"\nAction cancelled please make " + Fore.CYAN + "[tblXmain_transaction.csv]" + Fore.RED + " exists in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                    print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                    time.sleep(3)
                    self.exitFlag = True
                    return

            self.dividerSmall(title="Output Folder")

           
            if not os.path.exists(self.docPath.joinpath("Output")):
                if input(
                        Fore.RED + "[Output] folder not found. \n" + Style.RESET_ALL + "Do you want to create a folder? Y/N: ").lower() == "y":
                    os.mkdir((self.docPath.joinpath("Output")))
                    print(
                        Fore.GREEN + f"Folder [Output] created at: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Output')}" + Style.RESET_ALL)
                else:
                    print(
                        Fore.RED + f"\nAction cancelled please make sure this folder exists: \n" + Fore.CYAN + f"{self.docPath.joinpath('Output')}" + Style.RESET_ALL)
                    print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                    time.sleep(3)
                    self.exitFlag = True
                    return
            else:
                print(Fore.GREEN + "Output Folder Found." + Style.RESET_ALL)

            self.dividerSmall(title="dfClean Data")

           
            try:
                if reloadFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading dfClean...", "Loading dfClean Complete",
                                                       "Loading dfClean Failed",
                                                       0.05).start()

                fileName = directoryScanner("dfClean", self.docPath, Folder="Data", returnMethod="Last",
                                            loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                self.sourceObj = dataFrameLoader(fileName, self.docPath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                try:
                    self.loadingAnimator.stop(method="error")
                except:
                    pass
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}") 
                if not reloadFlag:
                    if input(
                            Fore.RED + "[dfClean] not found!!! " + Style.RESET_ALL + "Do you want to create this file? Y/N:").lower() == "y":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.CYAN + "[tblXmain_transactions.csv]" + Fore.YELLOW + " exists the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue" + " (Est. Time = 2 Minutes)..." + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Cleaning Data Set...", "Cleaning Complete",
                                                               "Cleaning Failed").start()
                        DataCleaner("tblXmain_transactions.csv",
                                    f"dfClean{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv", "Data",
                                    path=self.docPath, verbose=self.verboseFlagBool)
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\nAction cancelled please make " + Fore.CYAN + "[dfClean.csv]" + Fore.RED + " exists in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.GREEN + "Rebuilding CleanDF" + Fore.YELLOW + f"Make sure " + Fore.CYAN + "[tblXmain_transactions.csv]" + Fore.YELLOW + " exists the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                    if not self.skipFlag:
                        input("Press [enter] to continue" + " (Est. Time = 2 Minutes)..." + Style.RESET_ALL)
                    self.loadingAnimator = loadingAnimator("Cleaning Data Set...", "Cleaning Complete",
                                                           "Cleaning Failed").start()
                    fileName = directoryScanner("tblXmain_transactions", self.docPath, Folder="Data",
                                                returnMethod="Last", loadingAnimator=self.loadingAnimator,
                                                skipFlag=self.skipFlag)
                    DataCleaner(fileName, f"dfClean{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv", "Data",
                                path=self.docPath, verbose=self.verboseFlagBool)
                    self.loadingAnimator.stop()

            self.dividerSmall(title="dfDummy & dfTarget Data")

           
            try:
                if reloadFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading dfDummy and dfTarget...",
                                                       "Loading dfDummy and dfTarget Complete",
                                                       "Loading dfDummy and dfTarget Failed").start()
                fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                            loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                self.dummyObj = dataFrameLoader(fileName, self.docPath, "Data", verbose=self.verboseFlagBool)
                fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                            loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                self.targetObj = dataFrameLoader(fileName, self.docPath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                try:
                    self.loadingAnimator.stop(method="error")
                except:
                    pass
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}") 
                if not reloadFlag:
                    if input(
                            Fore.RED + "[dfDummy] or [dfTarget] not found!!! " + Style.RESET_ALL + f"Do you want to create this file? Y/N:").lower() == "y":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfClean.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue " + " (Est. Time = 1 Minute)..." + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Machine Learning Preparation...",
                                                               "Machine Learning Preparation Complete",
                                                               "Machine Learning Preparation Failed").start()
                        fileName = directoryScanner("dfClean", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        dataMLPrep(fileName, f"dfDummy{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv",
                                   f"dfTarget{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv", "Data",
                                   self.docPath,
                                   verbose=self.verboseFlagBool)
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Action cancelled please make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.RED + " and " + Fore.CYAN + "[dfTarget.csv]" + Fore.RED + " exist in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.GREEN + "Rebuilding dfDummy and dfTarget. " + Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfClean.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                    if not self.skipFlag:
                        input("Press [enter] to continue " + " (Est. Time = 1 Minute)..." + Style.RESET_ALL)
                    self.loadingAnimator = loadingAnimator("Machine Learning Preparation...",
                                                           "Machine Learning Preparation Complete",
                                                           "Machine Learning Preparation Failed").start()
                    fileName = directoryScanner("dfClean", self.docPath, Folder="Data", returnMethod="Last",
                                                loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                    dataMLPrep(fileName, f"dfDummy{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv",
                               f"dfTarget{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.csv", "Data",
                               self.docPath,
                               verbose=self.verboseFlagBool)
                    self.loadingAnimator.stop()

            self.dividerSmall(title="Scaling Model")

           
            try:
                if reloadFlag and retrainFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading Scaling Model...", "Loading Scaling Model Complete",
                                                       "Loading Scaling Model Failed").start()
                fileName = directoryScanner("scaler", self.docPath, Folder="Data", returnMethod="Last",
                                            loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                self.scalerObj = modelLoader(fileName, self.docPath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                try:
                    self.loadingAnimator.stop(method="error")
                except:
                    pass
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}") 
                if not reloadFlag and not retrainFlag:
                    self.loadingAnimator.stop()
                    if input(
                            Fore.RED + "[scaler.sav] not found or corrupted!!! " + Style.RESET_ALL + f"Do you want to create this file? Y/N:").lower() == "y":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue " + " (Est. Time = 30 Seconds)..." + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating Scaling Model...",
                                                               "Creating Scaling Model Complete",
                                                               "Creating Scaling Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Action cancelled please make " + Fore.CYAN + "[scaler.sav]" + Fore.RED + " exists in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.CYAN + "Scaler Model will be retrained in the next section.")
                    pass

            self.dividerSmall(title="Machine Learning Model")

           
            try:
                if reloadFlag and retrainFlag:
                    raise "Pass"

                self.loadingAnimator = loadingAnimator("Loading ML Model...", "Loading ML Model Complete",
                                                       "Loading ML Model Failed").start()
                fileName = directoryScanner("model", self.docPath, Folder="Data", returnMethod="Last",
                                            loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                self.modelObj = modelLoader(fileName, self.docPath, "Data", verbose=self.verboseFlagBool)
                self.loadingAnimator.stop()
            except Exception as e:
                try:
                    self.loadingAnimator.stop(method="error")
                except:
                    pass
                if self.DebugFlag:
                    print(Fore.RED + f"DEBUG MESSAGE::: {e}") 
                if not reloadFlag and not retrainFlag:
                    methodChoice = input(
                        Fore.RED + "[model.sav] not found or corrupted!!! " + Style.RESET_ALL + "\nDo you want to create this file using " + Fore.CYAN + "[O]" + Style.RESET_ALL + "ptimal, (Est. 4 Hours)" + Fore.CYAN + "[F]" + Style.RESET_ALL + "ast (Est. 5 min), or " + Fore.CYAN + "[C]" + Style.RESET_ALL + "ustom parameters? or " + Fore.CYAN + "[E]" + Style.RESET_ALL + "xit this action?:")
                    if methodChoice.lower() == "o":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.CYAN + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue or Ctrl+C to cancel " + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
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
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
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
                                f"Press [enter] to continue learning with" + Fore.CYAN + f"N_est = {n_estimatorsInp}" + Style.RESET_ALL + " and " +
                                Fore.CYAN + f"Max depth = {max_depthInp}" + Style.RESET_ALL + " or Ctrl+C to cancel...")

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
                                       ParamGrid=ParamGrid,
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Action cancelled please make " + Fore.CYAN + "[model.sav]" + Fore.RED + " exists in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return
                else:
                    print(
                        Fore.GREEN + "Rebuilding Scaler and Ensemble Models. " + Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfClean.csv]" + Fore.YELLOW + " exists in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                    methodChoice = input(
                        "Do you want to train the model with " + Fore.CYAN + "[O]" + Style.RESET_ALL + "ptimal, " + Fore.CYAN + "[F]" + Style.RESET_ALL + "ast, or " + Fore.CYAN + "[C]" + Style.RESET_ALL + "ustom parameters?")
                    if methodChoice.lower() == "o":
                        print(
                            Fore.YELLOW + f"Make sure " + Fore.CYAN + "[dfDummy.csv]" + Fore.YELLOW + " and " + Fore.CYAN + "[dfTarget.csv]" + Fore.YELLOW + " exist in the folder: \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        if not self.skipFlag:
                            input(
                                "Press [enter] to continue or Ctrl+C to cancel " + Style.RESET_ALL)

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
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
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
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
                                f"Press [enter] to continue learning with" + Fore.CYAN + f"N_est = {n_estimatorsInp}" + Style.RESET_ALL + " and " +
                                Fore.CYAN + f"Max depth = {max_depthInp}" + Style.RESET_ALL + " or Ctrl+C to cancel...")

                        self.loadingAnimator = loadingAnimator("Creating ML Model...",
                                                               "Creating ML Model Complete",
                                                               "Creating ML Model Failed").start()
                        fileName = directoryScanner("dfDummy", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        self.scalerObj = dataScaler(fileName,
                                                    f"scaler{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav",
                                                    self.docPath, "Data")
                        fileName = directoryScanner("dfTarget", self.docPath, Folder="Data", returnMethod="Last",
                                                    loadingAnimator=self.loadingAnimator, skipFlag=self.skipFlag)
                        machineLearner(self.scalerObj.scaledDf, fileName,
                                       f"model{datetime.datetime.today().strftime('%m%d%Y_%H%M%S')}.sav", self.docPath,
                                       ParamGrid=ParamGrid,
                                       Folder="Data")
                        self.loadingAnimator.stop()
                    else:
                        print(
                            Fore.RED + f"\n Wrong input detected" + Fore.CYAN + "[model.sav]" + Fore.RED + " exists in \n" + Fore.CYAN + f"  {self.docPath.joinpath('Data')}" + Style.RESET_ALL)
                        print(Fore.RED + "Closing Application in 3 seconds" + Style.RESET_ALL)
                        time.sleep(3)
                        self.exitFlag = True
                        return

            if reloadFlag:
                self.initFlag = True
            else:
                self.initFlag = False

    @staticmethod
    def dividerSmall(title=None, method="print", padding=None, fullLength=None):

        """
    The dividerSmall function is used to print a divider with the title of the section in it.
    The function takes three arguments:
        - title (str): The text that will be printed in the middle of the divider. If no text is provided, then only a line will be printed.
        - method (str): This argument determines whether or not to print or return the string containing all of our formatting and content for printing later on. Default value is &quot;print&quot;.
        - padding (int): This argument determines how much padding there should be between each side of our divider and where we want our content to start/

    Args:
        title: Set the text in the middle of the divider
        method: Determine whether to print the divider or return it
        padding: Add extra padding to the left and right of the title
        fullLength: Set the length of the divider

    Returns:
        A string

    Doc Author:
        Willem van der Schans, Trelent AI
    """
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
            print(Fore.MAGENTA + dividerText + Style.RESET_ALL)
        elif method.lower() == "return":
            return Fore.MAGENTA + dividerText + Style.RESET_ALL

    @staticmethod
    def asciiArt():
        """
            The asciiArt function returns a string containing the ASCII art for the Avid Acceptance logo.


            Args:

            Returns:
                A string of ascii art

            Doc Author:
                Willem van der Schans, Trelent AI
            """

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
        """
            The asciiArt function returns a string containing the small ASCII art for the Avid Acceptance logo.


            Args:

            Returns:
                A string of ascii art

            Doc Author:
                Willem van der Schans, Trelent AI
            """
        asciiArt = Fore.GREEN + """
            /\    
           /  \   
          / /\ \  
         / ____ \ 
        /_/    \_\
           Avid
        """ + Style.RESET_ALL

        return asciiArt


class loadingAnimator:
    def __init__(self, desc="Loading...", end="Done!", error="Loading Failed", timeout=0.3):
        """
    The __init__ function is called when the class is instantiated.
    It sets up the initial values of all variables and attributes that are needed for this class to function properly.
    The __init__ function takes in a description, an end message, an error message, and a timeout value as parameters.

    Args:
        self: Represent the instance of the class
        desc: Set the description of the loading bar
        end: Set the text that is displayed when the loading bar is finished
        error: Display an error message if the loading fails
        timeout: Set the time between each animation frame

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.desc = Fore.CYAN + desc + Style.RESET_ALL
        self.end = Fore.GREEN + end + Style.RESET_ALL
        self.timeout = timeout
        self.error = Fore.RED + error + Style.RESET_ALL
        self.startTime = time.time()

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = [Fore.CYAN + f"⢿" + Style.RESET_ALL,
                      Fore.CYAN + f"⣻" + Style.RESET_ALL,
                      Fore.CYAN + f"⣽" + Style.RESET_ALL,
                      Fore.CYAN + f"⣾" + Style.RESET_ALL,
                      Fore.CYAN + f"⣷" + Style.RESET_ALL,
                      Fore.CYAN + f"⣯" + Style.RESET_ALL,
                      Fore.CYAN + f"⣟" + Style.RESET_ALL,
                      Fore.CYAN + f"⡿" + Style.RESET_ALL]
        self.done = False

    def start(self):
        """
            The start function starts the thread.
            It also sets the startTime to time.time() so that we can keep track of how long it has been running.

            Args:
                self: Represent the instance of the class

            Returns:
                The object itself

            Doc Author:
                Willem van der Schans, Trelent AI
            """

        self._thread.start()
        self.startTime = time.time()
        return self

    def _animate(self):
        """
    The _animate function is a generator that cycles through the steps of the animation.
    It prints out each step, and then waits for self.timeout seconds before printing out the next step.

    Args:
        self: Make the class methods aware of other methods and attributes within the class

    Returns:
        A generator object

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c} T+ {datetime.timedelta(seconds=round(time.time() - self.startTime, 0))}",
                  flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        """
          The __enter__ function is called when the context manager is entered.
          It returns whatever object should be assigned to the variable in the as clause of a with statement.
          The __exit__ function is called when exiting from a with statement.

          Args:
              self: Access the class attributes

          Returns:
              The self object

          Doc Author:
              Willem van der Schans, Trelent AI
          """
        self.start()

    def stop(self, method=""):
        """
            The stop function is used to stop the progress bar.
            It can be called in three ways:
                1) No arguments, which will print a default message and then clear the line.
                2) A string argument, which will print that string and then clear the line.
                3) The keyword &quot;none&quot;, which will not print anything but still clear the line.

            Args:
                self: Represent the instance of the class
                method: Determine what message is printed when the bar is stopped

            Returns:
                A string that is either the error or end message

            Doc Author:
                Willem van der Schans, Trelent AI
            """

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
        """
          The __exit__ function is called when the context manager exits.
          It can be used to clean up resources, or handle exceptions that occurred in the with block.
          If an exception was raised in the with block, it will be passed as a parameter to __exit__.

          Args:
              self: Represent the instance of the class
              exc_type: Determine the type of exception that was thrown
              exc_value: Pass in the exception that was raised
              tb: Get the traceback object

          Returns:
              A boolean value

          Doc Author:
              Willem van der Schans, Trelent AI
          """
        self.stop()
