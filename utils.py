##########################################################################################################################
##  En este programa se definen funciones utiles para se llamadas desde
##  las interfaces o clases
##########################################################################################################################

import numpy as np
import pickle
import math

##  Cargar dataset en X y Y
def loadData(path, split = 0.8):

    with open(path, 'rb') as f:
        data = pickle.load(f)

    X, Y = zip(*data)

    X = np.array(X)
    Y = np.array(Y)

    X_train, Y_train, X_test, Y_test = splitDataset(X, Y, split)

    Y_train = toOneHot(Y_train)
    Y_test = toOneHot(Y_test)

    return X_train, Y_train, X_test, Y_test

def toOneHot(X):

    z = np.max(X) + 1

    X_oh = np.eye(z)[X]

    return X_oh

def splitDataset(X, Y, split = 0.8):

    m = X.shape[0]

    s = math.trunc(split*m)

    X_train = X[0:s,:,:]
    Y_train = Y[0:s]

    X_test = X[s:m,:,:]
    Y_test = Y[s:m]

    return X_train, Y_train, X_test, Y_test

