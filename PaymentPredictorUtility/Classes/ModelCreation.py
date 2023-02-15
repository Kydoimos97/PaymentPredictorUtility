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
#init(autoreset=True)



class dataScaler:

    def __init__(self, dataframeFileName, ModelFileName, path, Folder=None):

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
        column_names = list(self.df.keys())
        self.scaler_model = MinMaxScaler()
        dfScaled = self.scaler_model.fit_transform(self.df)
        self.scaledDf = pd.DataFrame(data=dfScaled, columns=column_names)

    def __modelExporter(self, model, filename):
        pickle.dump(model, open(f"{self.__Path.joinpath(filename)}", "wb"))


class machineLearner:

    def __init__(self, inputX, inputY, modelFile, path, method="xgBoost", Gridsearch=True, tableType=None,
                 ParamGrid=None, Folder=None, verbose=False):

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
        pickle.dump(model, open(f"{self.__Path.joinpath(modelFile)}", "wb"))

    def __testTrainCreator(self):
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(self.df, self.target,
                                                                                    test_size=0.25,
                                                                                    random_state=33)

    def __paramGridCreator(self, ParamGrid):

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
            # TODO Error
            raise ValueError

    def __xgboostTrainer(self, Gridsearch, ParamGrid, Path, Folder):

        if Gridsearch:

            self.__paramGridCreator(ParamGrid)

            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            counter = 0
            xgboost_timer_start = timeit.default_timer()
            optimalAucFold = 0
            verboseDict = {}

            for epoch in range(len(self.__paramGrid)):

                xgboost_timer_lap = timeit.default_timer()
                # Run Model
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

            # Rerun the model with optimal parameters
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

        if Gridsearch:
            __maxNeighbours = int(input("Max Neighbours [int]: "))
            __maxEpochs = int(input("Max Epochs to run [int]: "))
            __step_size = np.ceil(__maxNeighbours / __maxEpochs)

            __bestOos = 0
            __neighbours = 0

            for x in range(1, __maxNeighbours + 1, __step_size):
                __knn = KNeighborsClassifier(n_neighbors=x)
                __knn.fit(self.__xTrain, self.__yTrain)

                # In Sample
                self.PredOos = __knn.predict(self.__xTest)
                __accOos = accuracy_score(self.__yTest, self.PredOos)

                # Out of sample
                self.PredIs = __knn.predict(self.__xTrain)
                __accIs = accuracy_score(self.__yTrain, self.PredIs)

                # append to dict
                self.EpochAccOos.append(__accOos)
                self.EpochAccIs.append(__accIs)

                # Update Output
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

        if outOfSample:
            self.__confusionMatrix(self.PredOos, self.__yTest, returnOption="graph")
        else:
            self.__confusionMatrix(self.PredIs, self.__yTrain, returnOption="graph")

    @staticmethod
    def __confusionMatrix(yPrediction, yActual, returnOption="graph"):

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
        display(
            HTML("<div style='height: 400px; text-align: left;'>" + dataframe.style.render() + "</div>"))
