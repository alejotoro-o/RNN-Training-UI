##########################################################################################################################
##  En este programa se define la clase para crear el modelo clasificador
##  de gestos
##########################################################################################################################

from PyQt5 import QtGui, QtCore, QtWidgets
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.optimizers import Adam
from keras import backend as K

##  Bandera global para detener el entrenamiento
global stopTrainingFlag
stopTrainingFlag = False


##  Clase para enviar señales durante el entrenamiento
class Sender(QtCore.QObject):

    epoch = QtCore.pyqtSignal(int, float, float)

    def send(self, e, l, a):

        self.epoch.emit(e, l, a)

##  Objeto global para enviar señales
sender = Sender()


## Callback personalizado para detener el entrenamiento y enviar informacion al hilo principal
class stopTrainingCallback(Callback):

    def on_batch_end(self, batch, logs = None):

        if stopTrainingFlag:

            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs = None):

        sender.send(epoch, logs['loss'], logs['accuracy'])


##  Clase para construir y entrenar al modelo neuronal de reconocimiento de gestos
class GestureClassifier:

    ##  Constructor de la clase
    def __init__(self):
        pass

    ## Construir modelo
    def buildModel(self, architecture, time_steps = 200, channels = 4, n_classes = 8):
        
        X = Input(shape = (time_steps, channels))

        x = X

        idx = 0
        c = 0

        for item in architecture:
            if item['type'] == 'LSTM':
                idx += 1

        ##  Architecture es una lista de objetos que tiene el atributo 'type' que corresponde al tipo
        ##  de capa en la red y el atributo 'units' que son las unidades ocultas en cada capa
        ##  en Dropout no existe 'units' sino 'rate' que corresponde a la fraccion con que se
        ##  congelan las unidades ocultas

        for layer in architecture:

            if layer['type'] == 'LSTM':

                c +=1

                if c == idx:

                    x = LSTM(layer['units'], return_sequences = False)(x)

                else:

                    x = LSTM(layer['units'], return_sequences = True)(x)

            elif layer['type'] == 'Dense':

                x = Dense(layer['units'], activation = 'relu')(x)

            elif layer['type'] == 'Dropout':

                x = Dropout(layer['units'])(x)

        if (n_classes == 2):
            out = Dense(1, activation = 'sigmoid')(x)
        else:
            out = Dense(n_classes, activation = 'softmax')(x)

        self.model = Model(inputs = X, outputs = out)

    ##  Entrenar modelo
    def trainModel(self, X_train, Y_train, epochs = 100, batch_size = 32, n_classes = 8):

        ##  Parametros de configuracion
        if (n_classes == 2):
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']

        ## Callbacks en el entrenamiento
        model_checkpoint = ModelCheckpoint(filepath = str(pathlib.Path(__file__).parent.absolute()) + '/checkpoints/checkpoint.h5')
        stop = stopTrainingCallback()
        
        ##  Configuracion de los parametros de entrenamiento
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        ##  Inicio del entrenamiento
        self.model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, callbacks = [model_checkpoint, stop], verbose = 0)

    def trainExistingModel(self, model_path, X_train, Y_train):
        pass

    ##  Evaluar modelo
    def evaluate(self, X_test, Y_test):
        
        loss, acc = self.model.evaluate(X_test, Y_test, verbose = 0)

        return loss, acc

    ##  Realizar prediccion
    def predict(self):
        pass

    ##  Estructura del modelo
    def modelSummary(self):
        
        self.model.summary()

    ##  Guardar modelo
    def saveModel(self, path):
        
        self.model.save(path)

    def loadModel(self, path):

        self.model = load_model(path)
