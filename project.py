import statistics
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
import warnings


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    #colNames = ["X1 Relative Compactness", "X2 Surface Area", "X3 Wall Area",
    #            "X4 Roof Area", "X5 Overall Height", "X6 Orientation",
    #            "X7 Glazing Area", "X8 Glazing Area Distribution", "Y1 Heating Load", "Y2 Cooling Load"]

    # Διαβάζουμε τα δεδομένα του dataset.
    # https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset
    df = pd.read_csv("datasets/ENB2012_data.csv")

    X = df.drop(["Y1", "Y2"], axis = 1) # Το feature που θα χρησιμοποιηθεί για την πρόβλεψη του y.
    y = df[["Y1", "Y2"]] # Το target που θα προβλέψουμε.

    categories = ["Low", "Medium", "High"]

    # Διαχωρίζουμε τα δεδομένα σε training κ' testing sets. Χρησιμοποιούμε το 30% για testing κ' το 70% για training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    
    while True:
        inputData = pd.DataFrame([input().split()], columns = X.columns)

        prediction = clf.predict(inputData)[0]

        print(f"Heating load: {categories[prediction[0]-1]}")
        print(f"Cooling load: {categories[prediction[1]-1]}")

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))