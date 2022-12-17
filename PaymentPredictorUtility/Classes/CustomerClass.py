import pandas as pd
from sklearn.metrics import accuracy_score


class Customer:
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Attributes
    ----------
    acctrefno : int
        the customer accrefno identifier int[5]
    dfCleanName : String
        Cleaned Dataframe file name exported by DataCleaner() include .csv
    dfTargetName : String
        Target Dataframe file name exported by DataCleaner() include .csv
    dfDummiesName : String
        Dummies Dataframe file name exported by DataCleaner() include .csv
    Modelname : String
        Model file name exported by pickle in MachineLearner() include file exstention
    scalerName : String
    """

    def __init__(self, acctrefno, dfClean, dfTarget, dfDummies, Model, Scaler, Folder=None):
        # Customer Profile Variables
        self.acctrefno = acctrefno
        self.numberOfPayments = 0
        self.certainty = None
        self.nextPaymentProbability = None
        self.accuracy = None
        self.future = None
        self.paymentCodes = None
        self.paymentDates = None
        self.paymentPrediction = None

        # Local Variables
        self.__CustomerPred = None

        self.__target = None
        self.__df = None
        self.__target = None
        self.__dummy = None
        self.__Folder = None

        # Get data and store
        self.__df = self.__getData(dfClean)
        self.__dummy = self.__getData(dfDummies, Scaler, dfTarget, Mode="Dummy")

        # Get folder
        if Folder is None:
            self.__Folder = ""
        else:
            self.__Folder = Folder

        # Prediction of Current values
        self.__predFunc(Model)

        # Predict future
        self.__predFuture(Model)

    def getDF(self):
        df = self.__df
        df["prediction"] = self.__CustomerPred[1]
        return df

    def __getData(self, dfClean, Scaler=None, dfTarget=None, Mode=None):

        # Get customer data subset
        df = dfClean
        if Mode is None:
            Mode = ""

        if Mode.lower() == "dummy":
            if dfTarget is not None:
                df['target'] = dfTarget.iloc[:, 0]

            df = df.loc[df["acctrefno"] == int(self.acctrefno)]
            df = df.sort_values("payment_number", ascending=False)
            df.reset_index(drop=True, inplace=True)
            self.__target = df['target']
            df.drop(columns="target", axis=1, inplace=True)

            # Scaling
            column_names = list(df.keys())
            df = Scaler.transform(df)
            df = pd.DataFrame(data=df, columns=column_names)
        else:
            df = df.loc[df["acctrefno"] == self.acctrefno]
            self.paymentCodes = pd.Series(df['transaction_code'].unique())
            x = df.sort_values("payment_number").drop_duplicates('transaction_code', keep="last")
            self.paymentDates = pd.Series(x['date_due'])
            df = df.sort_values("payment_number", ascending=False)
            df.reset_index(drop=True, inplace=True)

        if self.numberOfPayments < max(df['payment_number']) or self.numberOfPayments is None:
            self.numberOfPayments = max(df['payment_number'])

        return df

    def __predFunc(self, Model):

        df = self.__dummy

        CustomerPredProb = Model.predict(df)
        CustomerAcc = pd.DataFrame(CustomerPredProb)
        self.__CustomerPred = CustomerAcc[0]
        CustomerAcc['Actual'] = self.__target
        self.accuracy = accuracy_score(CustomerAcc['Actual'], CustomerAcc[0])

    def __predFuture(self, Model):

        df = self.__dummy
        paymentNumberDistance = (max(df['payment_number']) - min(df['payment_number'])) / self.numberOfPayments
        # Generate future transactions
        df = df.sort_values("payment_number").drop_duplicates('transaction_code', keep="last")
        df['payment_number'] += paymentNumberDistance
        df.to_csv("Data/test2.csv")
        # Prediction
        CustomerPredProb = Model.predict_proba(df)
        CustomerAcc = pd.DataFrame(CustomerPredProb).rename(columns={0: "No Pay", 1: "Pay"})
        self.future = df
        self.certainty = round(sum(CustomerAcc["Pay"]) / len(CustomerAcc) * 100, 2)
        self.nextPaymentProbability = pd.Series(round(CustomerAcc['Pay'], 5))
        # Prediction Translation
        PredictionList = []
        for x in self.nextPaymentProbability.values:
            if x < 0.5:
                PredictionList.append(0)
            else:
                PredictionList.append(1)
        self.paymentPrediction = pd.Series(PredictionList)


