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
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import timeit

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import xgboost as xgb

from IPython.core.display import HTML
from IPython.core.display_functions import display
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

from colorama import Fore, init, Style
init(autoreset=True, convert=True)



class dataScaler:

    def __init__(self, dataframeFileName, ModelFileName, path, Folder=None):

        """
    The __init__ function is the first function that gets called when you create a new instance of a class.
    It's used to set up (initialize) any attributes or variables that the object will need.


    Args:
        self: Represent the instance of the class
        dataframeFileName: Import the dataframe
        ModelFileName: Save the model to a file
        path: Set the path of the dataframe and model files
        Folder: Specify a folder where the dataframe is located

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.scaledDf = None
        self.scaler_model = None

        self.scaledDf = None
        self.scaler_model = None

        self.__Path = Path(path)

        if Folder is None:
            pass
        else:
            self.__Path = self.__Path.joinpath(str(Folder))

        self.df = pd.read_csv(f"{self.__Path.joinpath(dataframeFileName)}", index_col=0)

        self.__scaler()
        self.__modelExporter(self.scaler_model, ModelFileName)

    def __scaler(self):
        """
    The __scaler function is a private function that takes the dataframe and scales it using MinMaxScaler.
    It then returns a scaled dataframe.

    Args:
        self: Represent the instance of the class

    Returns:
        A dataframe with scaled values

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        column_names = list(self.df.keys())
        self.scaler_model = MinMaxScaler()
        dfScaled = self.scaler_model.fit_transform(self.df)
        self.scaledDf = pd.DataFrame(data=dfScaled, columns=column_names)

    def __modelExporter(self, model, filename):
        """
    The __modelExporter function takes in a model and a filename, then exports the model to the specified file.


    Args:
        self: Allow an object to refer to itself inside of a method
        model: Pass the model to be saved
        filename: Specify the name of the file to be created

    Returns:
        A pickled model

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        pickle.dump(model, open(f"{self.__Path.joinpath(filename)}", "wb"))


class machineLearner:

    def __init__(self, inputX, inputY, modelFile, path, method="xgBoost", Gridsearch=True, tableType=None,
                 ParamGrid=None, Folder=None, verbose=False):

        """
    The __init__ function is the first function that gets called when you create an object.
    It sets up all of the variables and functions for your class.
    The self variable refers to the instance of a class, so it's like saying &quot;this&quot; in Java or C++.

    Args:
        self: Refer to the class itself
        inputX: Specify the input dataframe
        inputY: Specify the target variable
        modelFile: Save the model in a file
        path: Specify the path to the folder where all data is stored
        method: Determine which model to use
        Gridsearch: Choose whether or not to use gridsearch
        tableType: Determine the type of table that will be created
        ParamGrid: Pass a dictionary of parameters to the gridsearchcv function
        Folder: Specify the folder where the data is stored
        verbose: Print out the time it takes to run each function

    Returns:
        Nothing

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.ArgumentsEpoch = None
        self.EpochAccOos = None
        self.EpochAccIs = None
        self.PredIs = None
        self.PredOos = None
        self.Model = None

        self.Arguments = {}
        self.__modelName = modelFile
        self.__paramGrid = None
        self.__verbose = verbose
        self.__tableType = tableType

        self.__Path = Path(path)

        if Folder is None:
            pass
        else:
            self.__Path = self.__Path.joinpath(str(Folder))

        if type(inputX) == str:
            self.df = pd.read_csv(f"{self.__Path.joinpath(inputX)}", index_col=0)
        else:
            self.df = inputX
        if type(inputY) == str:
            self.target = pd.read_csv(f"{self.__Path.joinpath(inputY)}", index_col=0)
        else:
            self.target = inputY

        self.startTime = time.time()
        self.__testTrainCreator()

        if self.__verbose:
            print(f"Test train done in {round(time.time() - self.startTime, 5)} seconds")
        self.startTime = time.time()
        if method.lower() == "xgboost":
            self.__xgboostTrainer(Gridsearch=Gridsearch, ParamGrid=ParamGrid)
        elif method.lower() == "knn":
            self.__KNNTrainer(Gridsearch=Gridsearch)

    def __modelExporter(self, model, modelFile):
        """
    The __modelExporter function takes in a model and a file name, then exports the model to the specified file.


    Args:
        self: Represent the instance of the class
        model: Pass the model to be saved
        modelFile: Specify the name of the file that will be created to store our model

    Returns:
        A pickled model file

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        pickle.dump(model, open(f"{self.__Path.joinpath(modelFile)}", "wb"))

    def __testTrainCreator(self):
        """
    The __testTrainCreator function is a private function that creates the training and testing sets for the model.
    The train_test_split function from sklearn.model_selection is used to create these sets, with 25% of the data being
    used as test data and 75% of it being used as training data.

    Args:
        self: Allow an object to refer to itself inside of a class

    Returns:
        The training and testing data for the model

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(self.df, self.target,
                                                                                    test_size=0.25,
                                                                                    random_state=33)

    def __paramGridCreator(self, ParamGrid):

        """
    The __paramGridCreator function is used to create a parameter grid for the XGBoost model.
        The function takes in a dictionary of parameters and values, or one of two pre-defined grids:
            - &quot;optimal&quot; : A grid that contains the optimal parameters found by running an exhaustive search on all possible combinations.
            - &quot;fast&quot; : A grid that contains fast but not necessarily optimal parameters.

    Args:
        self: Represent the instance of the class
        ParamGrid: Define the grid of parameters to be used in the model

    Returns:
        A dataframe with all the possible combinations of parameters

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if type(ParamGrid) is dict:
            self.__paramGrid = pd.DataFrame(ParameterGrid(ParamGrid))
        elif ParamGrid.lower() == "optimal":
            paramGridInp = {"n_estimators": [2500],
                         "max_depth": [15],
                         "alpha": [1],
                         "learning_rate": [0.1],
                         "colsample_bytree": [0.8],
                         "lambda": [1],
                         "subsample": [1]}

            self.__paramGrid = pd.DataFrame(ParameterGrid(paramGridInp))

        elif ParamGrid.lower() == "fast":
            paramGridInp = {"n_estimators": [100],
                         "max_depth": [2],
                         "alpha": [1],
                         "learning_rate": [0.1],
                         "colsample_bytree": [0.8],
                         "lambda": [1],
                         "subsample": [1]}

            self.__paramGrid = pd.DataFrame(ParameterGrid(paramGridInp))
        else:
           
            raise ValueError

    def __xgboostTrainer(self, Gridsearch, ParamGrid, Path, Folder):

        """
    The __xgboostTrainer function is the main function that runs the xgboost model.
    It takes in a boolean value for Gridsearch, which determines whether or not to run a grid search over parameters.
    If Gridsearch is set to True, it will take in a dictionary of parameter values and create an exhaustive list of all possible combinations using __paramGridCreator().
    The function then uses KFold cross validation with 3 folds to train and test each combination on the training data.
    The mean accuracy score and AUC are calculated for each fold, then averaged across all folds for each epoch (combination).
    This process continues

    Args:
        self: Bind the object to the method
        Gridsearch: Determine whether the model should be run with a gridsearch or not
        ParamGrid: Define the grid of parameters to search over
        Path: Define the path to save the model
        Folder: Specify the folder to save the model in

    Returns:
        :

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if Gridsearch:

            self.__paramGridCreator(ParamGrid)

            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            counter = 0
            xgboost_timer_start = timeit.default_timer()
            optimalAucFold = 0
            verboseDict = {}

            for epoch in range(len(self.__paramGrid)):

                xgboost_timer_lap = timeit.default_timer()
               
                xgClassifier = xgb.XGBClassifier(objective='binary:logistic',
                                                 n_estimators=self.__paramGrid["n_estimators"][epoch],
                                                 max_depth=self.__paramGrid["max_depth"][epoch],
                                                 reg_alpha=self.__paramGrid["alpha"][epoch],
                                                 learning_rate=self.__paramGrid["learning_rate"][epoch],
                                                 colsample_bytree=self.__paramGrid["colsample_bytree"][epoch],
                                                 reg_lambda=self.__paramGrid["lambda"][epoch],
                                                 subsample=self.__paramGrid["subsample"][epoch],
                                                 verbosity=0,
                                                 use_label_encoder=False)

                accFold = []
                aucFold = []
                try:
                    for Train_index, Test_index in kf.split(self.__xTrain, self.__yTrain):
                        xTrainFold, xTestFold = self.__xTrain.iloc[Train_index], self.__xTrain.iloc[Test_index]
                        yTrainFold, yTestFold = self.__yTrain.iloc[Train_index], self.__yTrain.iloc[Test_index]
                        xgClassifier.fit(xTrainFold, yTrainFold)

                        yTestFoldPred = xgClassifier.predict(xTestFold)
                        accFold.append(accuracy_score(yTestFold, yTestFoldPred))
                        aucFold.append(roc_auc_score(yTestFold, yTestFoldPred))

                    meanFoldAcc = np.mean(accFold)
                    meanFoldAuc = np.mean(aucFold)

                except Exception as e:
                    raise Exception(e)

                if meanFoldAuc > optimalAucFold:
                    optimalAucFold = meanFoldAuc
                    self.Arguments = {"objective": 'binary:logistic',
                                      "tree_method": "gpu_hist",
                                      "gpu_id": 0,
                                      "fail_on_invalid_gpu_id": 1,
                                      "nthread": 5,
                                      "n_estimators": self.__paramGrid["n_estimators"][epoch],
                                      "max_depth": self.__paramGrid["max_depth"][epoch],
                                      "reg_alpha": self.__paramGrid["alpha"][epoch],
                                      "learning_rate": self.__paramGrid["learning_rate"][epoch],
                                      "colsample_bytree": self.__paramGrid["colsample_bytree"][epoch],
                                      "reg_lambda": self.__paramGrid["lambda"][epoch],
                                      "subsample": self.__paramGrid["subsample"][epoch],
                                      "verbose": 0}

                counter += 1
                xgboost_timer_stop = timeit.default_timer()

                if counter % 1 == 0 or counter == 1 or counter == len(self.__paramGrid):
                    if self.__verbose:
                        print(
                            f"Completed Epoch {counter}/{len(self.__paramGrid)} at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
                            f"time elapsed = {str(datetime.timedelta(seconds=round(xgboost_timer_stop - xgboost_timer_start, 0)))} | "
                            f"estimated time remaining {str(datetime.timedelta(seconds=round((xgboost_timer_stop - xgboost_timer_lap) * (len(self.__paramGrid) - counter), 0)))}")
                else:
                    pass

                epochParamGrid = {"Mean Fold Accuracy": meanFoldAcc,
                                  "Mean Fold Auc": meanFoldAuc,
                                  "n_estimators": self.__paramGrid["n_estimators"][epoch],
                                  "max_depth": self.__paramGrid["max_depth"][epoch],
                                  "reg_alpha": self.__paramGrid["alpha"][epoch],
                                  "learning_rate": self.__paramGrid["learning_rate"][epoch],
                                  "colsample_bytree": self.__paramGrid["colsample_bytree"][epoch],
                                  "reg_lambda": self.__paramGrid["lambda"][epoch],
                                  "subsample": self.__paramGrid["subsample"][epoch]}

                verboseDict[epoch + 1] = epochParamGrid
                self.ArgumentsEpoch = verboseDict

           
            xgClassifier = xgb.XGBClassifier(objective='binary:logistic',
                                             n_estimators=self.Arguments["n_estimators"],
                                             max_depth=self.Arguments["max_depth"],
                                             reg_alpha=self.Arguments["reg_alpha"],
                                             learning_rate=self.Arguments["learning_rate"],
                                             colsample_bytree=self.Arguments["colsample_bytree"],
                                             reg_lambda=self.Arguments["reg_lambda"],
                                             subsample=self.Arguments["subsample"],
                                             verbosity=0,
                                             use_label_encoder=False)
        else:
            __nEst = int(input("N_Estimators [int]: "))
            __maxDepth = int(input("Max Depth [int]: "))
            __regAlpha = int(input("Regulator Alpha [Float 0-1]: "))
            __LearningRate = int(input("Learning Rate [Float]: "))
            __regLambda = int(input("Regulator Lambda [FLoat 0-1]: "))

            xgClassifier = xgb.XGBClassifier(objective='binary:logistic',
                                             n_estimators=__nEst,
                                             max_depth=__maxDepth,
                                             reg_alpha=__regAlpha,
                                             learning_rate=__LearningRate,
                                             reg_lambda=__regLambda,
                                             verbosity=0,
                                             use_label_encoder=False)

            epochParamGrid = {"n_estimators": __nEst,
                              "max_depth": __maxDepth,
                              "reg_alpha": __regAlpha,
                              "learning_rate": __LearningRate,
                              "reg_lambda": __regLambda}

            self.ArgumentsEpoch = epochParamGrid

        xgClassifier.fit(self.__xTrain, self.__yTrain)
        if self.__verbose:
            print(f"fit done in {round(time.time() - self.startTime, 5)} seconds")
        self.startTime = time.time()
        self.PredOos = xgClassifier.predict(self.__xTest)
        outsampleAcc = accuracy_score(self.__yTest, self.PredOos)
        outsampleAUC = roc_auc_score(self.__yTest, self.PredOos)
        if self.__verbose:
            print(f"oos done in {round(time.time() - self.startTime, 5)} seconds")
        self.startTime = time.time()
        self.PredIs = xgClassifier.predict(self.__xTrain)
        insampleAcc = accuracy_score(self.__yTrain, self.PredIs)
        insampleAUC = roc_auc_score(self.__yTrain, self.PredIs)
        if self.__verbose:
            print(f"Is done in {round(time.time() - self.startTime, 5)} seconds")
        self.startTime = time.time()
        if self.__verbose:
            if Gridsearch:
                lineDF = pd.melt(pd.DataFrame(self.ArgumentsEpoch).transpose().reset_index(), id_vars=['index'],
                                 value_vars=["Mean Fold Accuracy", "Mean Fold Auc"]).rename(
                    columns={"variable": "Type"})
                self.__linePlot(lineDF, colName="value", hue="Type", xLabel="Epoch", yLabel="Score",
                                graphTitle="Fold Accuracy and Auc over epochs", figSize=(18, 6), setAxis=False)

            self.__barPlot(pd.DataFrame(self.PredIs).rename(columns={0: "Prediction"}), "Prediction")
            self.__confusionMatrix(self.PredOos, self.__yTest['paid'], returnOption="graph")

            if self.__tableType.lower() == "html":
                self.__printTableHtml(pd.DataFrame(self.ArgumentsEpoch).transpose())
            else:
                print(tabulate(pd.DataFrame(self.ArgumentsEpoch).transpose(), headers='keys', tablefmt='psql'))
        else:
            pass

        if self.__verbose:
            print(f"Graphs done in {round(time.time() - self.startTime, 5)} seconds")
        self.startTime = time.time()

        self.__modelExporter(xgClassifier, self.__modelName)
        if self.__verbose:
            print(f"Model Exporter done in {round(time.time() - self.startTime, 5)} seconds")
        self.startTime = time.time()

        self.__xTest['prediction'] = self.PredOos[0]
        self.__xTest.to_csv(f"{self.__Path.joinpath('test.csv')}")
        if self.__verbose:
            print(f"Test Exported in {time.time() - self.startTime} seconds")
        self.startTime = time.time()
        self.__xTrain['prediction'] = self.PredIs[0]
        self.__xTrain.to_csv(f"{self.__Path.joinpath('train.csv')}")
        if self.__verbose:
            print(f"Train Exported in {time.time() - self.startTime} seconds")
        self.startTime = time.time()

        result = f"\nIn-sample: Accuracy = " + Fore.CYAN + f"{round(insampleAcc,2)}" + Style.RESET_ALL + ", AUC = " + Fore.CYAN + f"{round(insampleAUC,2)}" + Style.RESET_ALL + " | Out-of-Sample: Accuracy = " + Fore.CYAN + f"{round(outsampleAcc,2)}" + Style.RESET_ALL + ", AUC = " + Fore.CYAN + f"{round(outsampleAUC,2)}" + Style.RESET_ALL + f"\n with parameters: {epochParamGrid}"
        result_bare = f"In-sample: Accuracy = {round(insampleAcc,2)}, AUC = {round(insampleAUC,2)} | Out-of-Sample: Accuracy = {round(outsampleAcc,2)}, AUC = {round(outsampleAUC,2)} \nwith parameters: {epochParamGrid}"
        file_path = Path(os.getcwd()).joinpath('Output').joinpath("Logs").joinpath(f"ModelLog{datetime.datetime.today().strftime('%Y%m%d_%H%M')}.log")

        if not os.path.exists(Path(os.getcwd()).joinpath('Output').joinpath("Logs")):
            os.mkdir(Path(os.getcwd()).joinpath('Output').joinpath("Logs"))

        with open(file_path, 'w') as fp:
            fp.write(result_bare)

        print(result)
        print(f"Log file saved to: " + Fore.CYAN + f"{file_path}")

    def __KNNTrainer(self, Gridsearch):

        """
    The __KNNTrainer function is a private function that trains the KNN model.
        It takes in two arguments: Gridsearch and verbose.

        If Gridsearch is True, it will run a grid search to find the best number of neighbours for the model.
            The user will be prompted to enter an integer value for maxNeighbours and maxEpochs (the maximum number of epochs).

    Args:
        self: Access the attributes and methods of a class
        Gridsearch: Determine whether the user wants to run a gridsearch or not

    Returns:
        A model and the accuracy of the model

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if Gridsearch:
            __maxNeighbours = int(input("Max Neighbours [int]: "))
            __maxEpochs = int(input("Max Epochs to run [int]: "))
            __step_size = np.ceil(__maxNeighbours / __maxEpochs)

            __bestOos = 0
            __neighbours = 0

            for x in range(1, __maxNeighbours + 1, __step_size):
                __knn = KNeighborsClassifier(n_neighbors=x)
                __knn.fit(self.__xTrain, self.__yTrain)

               
                self.PredOos = __knn.predict(self.__xTest)
                __accOos = accuracy_score(self.__yTest, self.PredOos)

               
                self.PredIs = __knn.predict(self.__xTrain)
                __accIs = accuracy_score(self.__yTrain, self.PredIs)

               
                self.EpochAccOos.append(__accOos)
                self.EpochAccIs.append(__accIs)

               
                if __accOos > __bestOos:
                    if __accIs > .95:
                        pass
                    else:
                        __bestOos = __accOos
                        __neighbours = x
        else:
            __neighbours = int(input("Neighbours [int]: "))

        __knn = KNeighborsClassifier(n_neighbors=__neighbours)
        __knn.fit(self.__xTrain, self.__yTrain)

        self.PredIs = __knn.predict(self.__xTrain)
        self.PredOos = __knn.predict(self.__xTest)
        self.AccDict = {"Is": round(accuracy_score(self.__yTrain, self.PredIs) * 100, 2),
                        "Oos": round(accuracy_score(self.__yTest, self.PredOos) * 100, 2)}
        self.Arguments = {'n_neighbors': __neighbours}
        self.Model = __knn

        if self.__verbose:
            if Gridsearch:
                meltDF = pd.DataFrame({"Out-of-Sample": self.EpochAccOos, "In-Sample": self.EpochAccIs}).reset_index()
                meltDF = pd.melt(meltDF, ['index']).rename(columns={"variable": "Type"})
                meltDF["index"] += 1
                self.__linePlot(meltDF, colName="value", hue="Type", xLabel="Neighbours", yLabel="Accuracy",
                                graphTitle="In-Sample vs Out-of-Sample scores")

            self.__barPlot(pd.DataFrame(self.PredIs).rename(columns={0: "Prediction"}), "Prediction")
            self.__confusionMatrix(self.PredOos, self.__yTest['paid'], returnOption="graph")
            print(
                f"The accuracy of the model with {__neighbours} Neighbours is {round(accuracy_score(self.__yTest, self.PredOos) * 100, 2)}% for out-of-sample and {round(accuracy_score(self.__yTrain, self.PredIs) * 100, 2)}% for in-sample")

        else:
            pass

        self.__modelExporter(__knn, self.__modelName)

    def __confusionMatrixShow(self, outOfSample=True):

        """
    The __confusionMatrixShow function is a helper function that displays the confusion matrix for either the in-sample or out-of-sample predictions.

    Args:
        self: Bind the method to a class
        outOfSample: Determine whether the confusion matrix is based on the in-sample or out-of-sample data

    Returns:
        A graph of the confusion matrix

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        if outOfSample:
            self.__confusionMatrix(self.PredOos, self.__yTest, returnOption="graph")
        else:
            self.__confusionMatrix(self.PredIs, self.__yTrain, returnOption="graph")

    @staticmethod
    def __confusionMatrix(yPrediction, yActual, returnOption="graph"):

        """
    The __confusionMatrix function takes in the predicted and actual values of a classification model,
    and returns either a dataframe or graph of the confusion matrix. The function is called by using:
        __confusionMatrix(yPrediction, yActual)
    where yPrediction is an array containing all predictions made by your model, and yActual contains all actual values.
    The default return option for this function is &quot;graph&quot;, which will display a heatmap of the confusion matrix. If you would like to return only the dataframe version of this object instead, use:
        __confusionMatrix(yPrediction

    Args:
        yPrediction: Predict the actual values of yactual
        yActual: Store the actual values of the data
        returnOption: Return either a dataframe or a graph

    Returns:
        A graph of the confusion matrix

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        plt.close("all")

        plt.rcParams.update({
            "lines.color": "white",
            "patch.edgecolor": "white",
            "text.color": "black",
            "axes.facecolor": "white",
            "axes.edgecolor": "lightgray",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "lightgray",
            "figure.facecolor": "black",
            "figure.edgecolor": "black",
            "savefig.facecolor": "black",
            "savefig.edgecolor": "black"})

        confusion_matrix = pd.crosstab(yActual, yPrediction, rownames=['Actual'], colnames=['Predicted'])

        if returnOption.lower() == "dataframe":
            return pd.DataFrame(confusion_matrix)
        elif returnOption.lower() == "graph":
            group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.values.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in
                                 confusion_matrix.values.flatten() / np.sum(confusion_matrix.values)]

            labels = [f"T: {v1}\n N: {v2} \n W: {v3}" for v1, v2, v3 in
                      zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2, 2)

            sns.set(rc={'figure.figsize': (6, 4), "font.size": 10, "axes.titlesize": 15, "axes.labelsize": 10,
                        'xtick.labelsize': 7, 'ytick.labelsize': 7})
            sns.set_style('whitegrid')
            ax = sns.heatmap(confusion_matrix, annot=labels, fmt='',
                             cmap=sns.color_palette(["#23ec12", "#0d5d06", "#23ec12", "#0d5d06"]), linewidths=0.5,
                             linecolor="white", cbar=False,
                             annot_kws={'fontstyle': 'italic', 'color': 'white', 'alpha': 1})
            ax.set_title("Confusion Matrix of Actual vs Predicted")
            plt.show()

    @staticmethod
    def __linePlot(dataFrame, colName, hue=None, graphTitle=None, xLabel=None, yLabel=None, figSize=None,
                   setAxis=True,
                   splitLine=False):

        """
    The __linePlot function is a helper function that plots the data in a line plot.
        It takes in the following parameters:
            - dataFrame: The pandas DataFrame containing the data to be plotted.
            - colName: The name of the column whose values are to be plotted on y-axis.
                This parameter is required and must be passed as string value, else an error will occur.
            - hue (optional): A categorical variable that will produce points with different colors for each level of hue,
                or None if no grouping by color should happen (default). If this parameter is

    Args:
        dataFrame: Pass the dataframe to be used in the function
        colName: Specify the column name of the dataframe that is to be plotted
        hue: Split the data into different lines
        graphTitle: Set the title of the graph
        xLabel: Set the label of the x-axis
        yLabel: Set the label of the y axis
        figSize: Set the size of the graph
        setAxis: Set the x-axis ticks
        splitLine: Split the line plot in two parts

    Returns:
        A line plot of the dataframe

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        plt.close("all")

        if graphTitle is None:
            graphTitle = f"Count plot of {colName}"
        if xLabel is None:
            xLabel = "index"
        if yLabel is None:
            yLabel = colName
        if figSize is None:
            figSize = (6, 4)

        if splitLine:
            concatDf = pd.DataFrame({"index": list(range(max(dataFrame["index"]))),
                                     hue: ["Guide"] * max(dataFrame['index']),
                                     colName: [((max(dataFrame[colName]) - min(dataFrame[colName])) / 2) + min(
                                         dataFrame[colName])] * max(dataFrame['index'])})

            dataFrame = pd.concat([dataFrame, concatDf])

        sns.set(rc={'figure.figsize': figSize, "font.size": 10, "axes.titlesize": 15, "axes.labelsize": 10,
                    'xtick.labelsize': 7, 'ytick.labelsize': 7})
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=dataFrame, palette=sns.color_palette(["#23ec12", "#0d5d06", "#787878"]), x="index",
                          y=colName, hue=hue)
        ax.set(xlabel=xLabel, ylabel=yLabel)
        if setAxis:
            ax.set(xticks=np.arange(min(dataFrame["index"]), max(dataFrame["index"]) + 1, 1))
            if max(dataFrame["index"] > 20):
                ax.xaxis.set_major_locator(ticker.MultipleLocator(int(max(dataFrame["index"] / 10))))
        else:
            pass
        ax.set_title(label=graphTitle)
        plt.show()

    @staticmethod
    def __barPlot(dataFrame, colName, graphTitle=None):
        """
    The __barPlot function is a helper function that takes in a dataFrame, column name and graph title.
    It then plots the count of each unique value in the specified column as a bar plot.
    The default graph title is &quot;Count plot of {colName}&quot;. The color palette used for this function is green.

    Args:
        dataFrame: Pass the dataframe to be used in the function
        colName: Specify the column name of the dataframe that will be plotted
        graphTitle: Set the title of the graph

    Returns:
        A bar plot of the values in a column

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        plt.close("all")

        if graphTitle is None:
            graphTitle = f"Count plot of {colName}"
        else:
            pass

        sns.set(rc={'figure.figsize': (6, 4), "font.size": 10, "axes.titlesize": 15, "axes.labelsize": 10,
                    'xtick.labelsize': 7, 'ytick.labelsize': 7})
        sns.set_style('whitegrid')
        ax = sns.barplot(data=dataFrame[colName].value_counts().reset_index(),
                         palette=sns.color_palette(["#23ec12", "#0d5d06"]), x="index", y=colName, saturation=1)
        ax.set_title(label=graphTitle)
        ax.bar_label(ax.containers[0])

        plt.show()

    @staticmethod
    def __printTableHtml(dataframe):
        """
    The __printTableHtml function is a helper function that takes in a pandas dataframe and prints it out as an HTML table.
    This is useful for displaying the results of our queries in Jupyter Notebook.

    Args:
        dataframe: Pass the dataframe to be displayed in the table

    Returns:
        The html table of the dataframe

    Doc Author:
        Willem van der Schans, Trelent AI
    """
        display(
            HTML("<div style='height: 400px; text-align: left;'>" + dataframe.style.render() + "</div>"))
