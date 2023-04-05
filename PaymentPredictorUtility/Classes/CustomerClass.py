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

import pandas as pd
from sklearn.metrics import accuracy_score


class Customer:

    def __init__(self, acctrefno, dfClean, dfTarget, dfDummies, Model, Scaler, Folder=None):
       
        """
    The __init__ function is the first function that gets called when you create a new instance of a class.
    It's job is to initialize all of the attributes of an object.
    The self parameter refers to the current instance of an object, and allows us to access variables that belong
    to classes

    Args:
        self: Represent the instance of the class
        acctrefno: Identify the customer
        dfClean: Get the data from the database
        dfTarget: Get the target dataframe
        dfDummies: Get the dummy variables for the customer
        Model: Pass the model to the class
        Scaler: Scale the data
        Folder: Store the results of the prediction

    Returns:
        The object itself, which is assigned to the variable self

    Doc Author:
        Trelent
    """
        self.acctrefno = acctrefno
        self.numberOfPayments = 0
        self.certainty = None
        self.nextPaymentProbability = None
        self.accuracy = None
        self.future = None
        self.paymentCodes = None
        self.paymentDates = None
        self.paymentPrediction = None

       
        self.__CustomerPred = None

        self.__target = None
        self.__df = None
        self.__target = None
        self.__dummy = None
        self.__Folder = None

       
        self.__df = self.__getData(dfClean)
        self.__dummy = self.__getData(dfDummies, Scaler, dfTarget, Mode="Dummy")

       
        if Folder is None:
            self.__Folder = ""
        else:
            self.__Folder = Folder

       
        self.__predFunc(Model)

       
        self.__predFuture(Model)

    def getDF(self):
        """
    The getDF function returns a dataframe with the following columns:
        - CustomerID
        - Gender
        - Age (in years)
        - Annual Income (in thousands of dollars)
        - Spending Score (0-100, where 100 is most likely to spend money in mall)

    Args:
        self: Represent the instance of the class

    Returns:
        A dataframe with the prediction column added

    Doc Author:
        Trelent
    """
        df = self.__df
        df["prediction"] = self.__CustomerPred[1]
        return df

    def __getData(self, dfClean, Scaler=None, dfTarget=None, Mode=None):

       
        """
    The __getData function is used to get the data for a specific customer.
    It takes in the following parameters:
        dfClean - The cleaned dataset that has been preprocessed and scaled.
        Scaler - The scaler object used to scale the data (if needed).  This is only required if Mode = &quot;Dummy&quot;.
        dfTarget - A target variable, which can be either a binary or multiclass classification problem.  This is only required if Mode = &quot;Dummy&quot;.
        Mode - Either None, Dummy, or Test.  If None then it will return all of the payments for a given customer

    Args:
        self: Access the instance of the class
        dfClean: Pass the dataframe to be cleaned
        Scaler: Scale the data
        dfTarget: Pass the target dataframe to the function
        Mode: Determine whether the function is being called in a training or prediction context

    Returns:
        The dataframe for the customer with acctrefno=self

    Doc Author:
        Trelent
    """
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

        """
    The __predFunc function takes in a model and returns the predicted probabilities of churn for each customer.
    It also calculates the accuracy score of the model.

    Args:
        self: Represent the instance of the class
        Model: Pass the model that is being used to predict

    Returns:
        The predicted probability of the customers who are likely to churn

    Doc Author:
        Trelent
    """
        df = self.__dummy

        CustomerPredProb = Model.predict(df)
        CustomerAcc = pd.DataFrame(CustomerPredProb)
        self.__CustomerPred = CustomerAcc[0]
        CustomerAcc['Actual'] = self.__target
        self.accuracy = accuracy_score(CustomerAcc['Actual'], CustomerAcc[0])

    def __predFuture(self, Model):

        """
    The __predFuture function is used to predict the future payment behavior of a customer.
        The function takes in a model and uses it to predict the probability that each transaction will be paid.
        It then calculates the certainty of all payments being made, as well as an overall prediction for whether or not
        each transaction will be paid.

    Args:
        self: Bind the method to an object
        Model: Pass the model to the function

    Returns:
        The future payment probability and the prediction of the next payment

    Doc Author:
        Trelent
    """
        df = self.__dummy
        paymentNumberDistance = (max(df['payment_number']) - min(df['payment_number'])) / self.numberOfPayments
       
        df = df.sort_values("payment_number").drop_duplicates('transaction_code', keep="last")
        df['payment_number'] += paymentNumberDistance
       
        CustomerPredProb = Model.predict_proba(df)
        CustomerAcc = pd.DataFrame(CustomerPredProb).rename(columns={0: "No Pay", 1: "Pay"})
        self.future = df
        self.certainty = round(sum(CustomerAcc["Pay"]) / len(CustomerAcc) * 100, 2)
        self.nextPaymentProbability = pd.Series(round(CustomerAcc['Pay'], 5))
       
        PredictionList = []
        for x in self.nextPaymentProbability.values:
            if x < 0.5:
                PredictionList.append(0)
            else:
                PredictionList.append(1)
        self.paymentPrediction = pd.Series(PredictionList)


