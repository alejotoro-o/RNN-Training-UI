##########################################################################################################################
##  Este es el programa principal donde se encuentra toda la logica,
##  permite aislar el codigo de las interfaces graficas
##########################################################################################################################

import pathlib
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from trainUi import Ui_trainWindow
from trainingConfigDialog import Ui_trainingConfigDialog
import gestureClassifier
from gestureClassifier import GestureClassifier
from utils import loadData
from keras import backend as K

##########################################################################################################################
##  Objeto para invocar un nuevo hilo          
##########################################################################################################################

class Worker(QtCore.QObject):

    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    test = QtCore.pyqtSignal(float, float)

    def __init__(self, X_train, Y_train, X_test, Y_test, config, parent = None):

        super(Worker, self).__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.config = config

    def train(self):

        self.started.emit()

        try:

            if self.config['pretrainedModel']:
                
                g = GestureClassifier()
                g.loadModel(self.config['pretrainedModelPath'])

            else:

                g = GestureClassifier()
                g.buildModel(self.config['architecture'], time_steps = self.config['timeSteps'], channels = self.config['channels'], n_classes = self.config['classes'])

            g.modelSummary()
            g.trainModel(self.X_train, self.Y_train, epochs = self.config['epochs'], batch_size = self.config['batch'], n_classes = self.config['classes'])

            gestureClassifier.stopTrainingFlag = False

            loss, acc = g.evaluate(self.X_test, self.Y_test)
            self.test.emit(loss, acc)

            g.saveModel(self.config['modelPath'])

            K.clear_session()

        except:

            print('Ha ocurrido un error')

        self.finished.emit()



##  Subclase de la pagina entrenamiento de red neuronal
class TrainWindow(QtWidgets.QMainWindow, Ui_trainWindow):

    def __init__(self, parent = None):

        super(TrainWindow, self).__init__(parent)
        self.setupUi(self)

        self.openIndexWindow.clicked.connect(self.hide)

        self.stopTraining.setEnabled(False)
        self.startTraining.setEnabled(False)
        self.openIndexWindow.setVisible(False)

        self.lossGraph.setTitle("Costo vs. Epoch", color = 'k', size = '16pt')
        self.lossGraph.setBackground('w')
        self.lossGraph.setLabel('left', 'Costo', **{'color': 'blue', 'font-size': '10pt'})
        self.lossGraph.setLabel('bottom', 'Epoch', **{'color': 'blue', 'font-size': '10pt'})
        self.lossGraph.showGrid(x = True, y = True)

        self.accuracyGraph.setTitle("Exactitud vs. Epoch", color = 'k', size = '16pt')
        self.accuracyGraph.setBackground('w')
        self.accuracyGraph.setLabel('left', 'Exactitud', **{'color': 'blue', 'font-size': '10pt'})
        self.accuracyGraph.setLabel('bottom', 'Epoch', **{'color': 'blue', 'font-size': '10pt'})
        self.accuracyGraph.showGrid(x = True, y = True)

        ##  Configuracion de red neuronal por defecto
        self.config = {
            'pretrainedModel': False,
            'pretrainedModelPath': '',
            'dataPath': '',
            'modelPath': str(pathlib.Path(__file__).parent.absolute()) + '/models/my_model.h5',
            'epochs': 100,
            'batch': 32,
            'classes': 4,
            'timeSteps': 200,
            'channels': 4,
            'architecture': [
                {'type': 'LSTM', 'units': 8}
            ]
        }
        
        self.epochs = []
        self.accuracy = []
        self.losses = []

        ##  Señal para abrir dialogo de configuracion de entrenamiento
        self.openTrainingConfig.clicked.connect(self.openDialog)

        ##  Señal para comenzar entrenamiento
        self.startTraining.clicked.connect(self.train)
        self.stopTraining.clicked.connect(self.stop)

        self.trainProgress.setValue(0)
        self.trainProgress.setMaximum(self.config['epochs'] - 1)

        gestureClassifier.sender.epoch.connect(self.updateProgress)

    ##  Actualiza informacion sobre el estado del entrenamiento
    def updateProgress(self, e, l, a):

        self.epochs.append(e)
        self.losses.append(l)
        self.accuracy.append(a)

        self.trainProgress.setValue(e)

        self.lossGraph.plot(self.epochs, self.losses, pen = pg.mkPen('r'))
        self.accuracyGraph.plot(self.epochs, self.accuracy, pen = pg.mkPen('r'))

        self.epochLabel.setText("Epoch: {}/{}".format(e, self.config['epochs'] - 1))
        self.lossDisp.setText("Costo de entrenamiento: {:7.4f}".format(l))
        self.accuracyDisp.setText("Exactitud de entrenamiento: {:7.4f}".format(a))


    ##  Abre el dialogo de configuracion y administra sus señales
    def openDialog(self):

        configDialog = TrainingConfig(self)
        configDialog.open()

        configDialog.accepted.connect(lambda: self.configSaved(configDialog))
        configDialog.rejected.connect(self.configCanceled)

    ##  Guarda la configuracion establecida en el cuadro de dialogo
    def configSaved(self, d):

        self.config = d.config

        self.trainProgress.setMaximum(self.config['epochs'] - 1)
        self.trainProgress.setValue(0)

        self.startTraining.setEnabled(True)

    def configCanceled(self):

        self.config = {
            'pretrainedModel': False,
            'pretrainedModelPath': '',
            'dataPath': '',
            'modelPath': str(pathlib.Path(__file__).parent.absolute()) + '/models/my_model.h5',
            'epochs': 100,
            'batch': 32,
            'classes': 4,
            'timeSteps': 200,
            'channels': 4,
            'architecture': [
                {'type': 'LSTM', 'units': 8}
            ]
        }

    ##  Inicia un hilo para el entrenamiento de la red neuronal
    def train(self):

        try:

            self.epochs = []
            self.accuracy = []
            self.losses = []

            self.lossGraph.clear()
            self.accuracyGraph.clear()

            X_train, Y_train, X_test, Y_test = loadData(self.config['dataPath'])
            
            self.worker = Worker(X_train, Y_train, X_test, Y_test, self.config)
            self.thread = QtCore.QThread()

            self.worker.moveToThread(self.thread)

            self.startTraining.setEnabled(False)
            self.stopTraining.setEnabled(True)
            self.openIndexWindow.setEnabled(False)
            self.openTrainingConfig.setEnabled(False)

            self.worker.test.connect(self.test)

            ##  Señales de finalizacion del entrenamiento
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(lambda: self.startTraining.setEnabled(True))
            self.worker.finished.connect(lambda: self.stopTraining.setEnabled(False))
            self.worker.finished.connect(lambda: self.openIndexWindow.setEnabled(True))
            self.worker.finished.connect(lambda: self.openTrainingConfig.setEnabled(True))

            self.thread.started.connect(self.worker.train)

            self.thread.start()

        except:

            print('Ha ocurrido un error')

    def test(self, l, a):

        self.testDisp.setText("Costo de prueba: {:7.4f} - Exactitud de prueba: {:7.4f}".format(l, a))

    ##  Sirve para detener el entrenamiento de la red neuronal
    def stop(self):
        
        gestureClassifier.stopTrainingFlag = True

        self.startTraining.setEnabled(True)
        self.stopTraining.setEnabled(False)
        self.openIndexWindow.setEnabled(True)
        self.openTrainingConfig.setEnabled(True)

        self.thread.quit()

    ##  Advierte al usuario de que el entrenamiento se detendra si se cierra el programa
    def closeEvent(self, event):

        try:
            if self.thread.isRunning():

                quit_msg = "Esta seguro que desea cerrar el programa? El entrenamiento sera detenido y su modelo se guardara."
                reply = QtWidgets.QMessageBox.warning(self, 'Advertencia!', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.Yes:
                    self.stop()
                    event.accept()
                else:
                    event.ignore()

            else:
                event.accept()
                
        except:
            event.accept()




##  Subclase del dialogo para la configuracion del entrenamiento y red neuronal
class TrainingConfig(QtWidgets.QDialog, Ui_trainingConfigDialog):

    def __init__(self, parent = None):

        super(TrainingConfig, self).__init__(parent)
        self.setupUi(self)

        self.config = parent.config

        self.pretrainedModelPath.setEnabled(False)
        self.dropoutRate.setEnabled(False)

        if self.config['pretrainedModel']:
            self.pretrainedModelPath.setText(self.config['pretrainedModelPath'])
            self.pretrainedModelCheck.setChecked(True)
            self.loadPretrainedModel.setEnabled(True)
            self.pretrainedModelPath.setEnabled(True)
            self.architectureTab.setEnabled(False)
        else:
            self.pretrainedModelCheck.setChecked(False)
            self.loadPretrainedModel.setEnabled(False)
            self.pretrainedModelPath.setEnabled(False)
            self.architectureTab.setEnabled(True)

        self.dataPath.setText(self.config['dataPath'])
        self.modelPath.setText(self.config['modelPath'])
        self.epochNumber.setValue(self.config['epochs'])
        self.batchSize.setValue(self.config['batch'])

        ##  Controles del cuadro de dialogo
        self.cancelConfig.clicked.connect(self.reject)
        self.saveConfig.clicked.connect(self.accept)

        self.loadDataPath.clicked.connect(self.getDataPath)
        self.saveModelPath.clicked.connect(self.setModelPath)
        self.loadPretrainedModel.clicked.connect(self.getPretrainedmodelPath)

        self.pretrainedModelCheck.stateChanged.connect(self.checkPretrainedModel)

        self.epochNumber.valueChanged.connect(self.setEpochNumber)
        self.batchSize.valueChanged.connect(self.setBatchSize)

        self.layerType.activated.connect(self.layerTypeSelection)

        self.addLayer.clicked.connect(self.addRow)
        self.removeLayer.clicked.connect(self.deleteRow)
        self.timeSteps.valueChanged.connect(self.updateIO)
        self.channels.valueChanged.connect(self.updateIO)
        self.classes.valueChanged.connect(self.updateIO)

        ##  Inicializacion de la tabla de estructura de la red neuronal
        self.architectureTable.insertRow(0)
        self.architectureTable.setItem(0, 0, QtGui.QTableWidgetItem('Entrada'))
        self.architectureTable.setItem(0, 1, QtGui.QTableWidgetItem('(' + str(self.config['timeSteps']) + ', ' + str(self.config['channels']) + ')'))

        for l in self.config['architecture']:

            rowCount = self.architectureTable.rowCount()
            self.architectureTable.insertRow(rowCount)

            self.architectureTable.setItem(rowCount, 0, QtGui.QTableWidgetItem(l['type']))
            self.architectureTable.setItem(rowCount, 1, QtGui.QTableWidgetItem(str(l['units'])))

        rowCount = self.architectureTable.rowCount()
        self.architectureTable.insertRow(rowCount)
        self.architectureTable.setItem(rowCount, 0, QtGui.QTableWidgetItem('Salida'))
        self.architectureTable.setItem(rowCount, 1, QtGui.QTableWidgetItem(str(self.config['classes'])))



    def getDataPath(self):

        dataPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Pickle files (*.pickle *.pkl)")
        self.config['dataPath'] = dataPath
        self.dataPath.setText(dataPath)


    def setModelPath(self):

        modelPath, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Open file', 'c:\\', "HDF5 files (*.h5)")
        self.config['modelPath'] = modelPath
        self.modelPath.setText(modelPath)

    def checkPretrainedModel(self):

        if self.pretrainedModelCheck.isChecked():

            self.config['pretrainedModel'] = True
            self.pretrainedModelPath.setEnabled(True)
            self.loadPretrainedModel.setEnabled(True)
            self.architectureTab.setEnabled(False)

        else:

            self.config['pretrainedModel'] = False
            self.pretrainedModelPath.setEnabled(False)
            self.loadPretrainedModel.setEnabled(False)
            self.architectureTab.setEnabled(True)

    def getPretrainedmodelPath(self):

        pretrainedmodelPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "HDF5 files (*.h5 *.hdf5)")
        self.config['pretrainedModelPath'] = pretrainedmodelPath
        self.pretrainedModelPath.setText(pretrainedmodelPath)

    def setEpochNumber(self):
        
        self.config['epochs'] = self.epochNumber.value()

    def setBatchSize(self):
        
        self.config['batch'] = self.batchSize.value()


    def layerTypeSelection(self, idx):

        if self.layerType.itemText(idx) == 'Dropout':

            self.dropoutRate.setEnabled(True)
            self.hiddenUnits.setEnabled(False)

        else:

            self.dropoutRate.setEnabled(False)
            self.hiddenUnits.setEnabled(False)

    def updateIO(self):

        self.config['timeSteps'] = self.timeSteps.value()
        self.config['channels'] = self.channels.value()
        self.config['classes'] = self.classes.value()
        self.architectureTable.setItem(0, 1, QtGui.QTableWidgetItem('(' + str(self.config['timeSteps']) + ', ' + str(self.config['channels']) + ')'))
        self.architectureTable.setItem(self.architectureTable.rowCount() - 1, 1, QtGui.QTableWidgetItem(str(self.config['classes'])))
            
    def addRow(self):

        if self.architectureTable.currentRow() == -1 or self.architectureTable.currentRow() == self.architectureTable.rowCount() - 1:

            rowCount = self.architectureTable.rowCount()
            self.architectureTable.insertRow(rowCount - 1)
            self.architectureTable.setItem(rowCount - 1, 0, QtGui.QTableWidgetItem(self.layerType.currentText()))
            if self.layerType.currentText() == 'Dropout':
                self.architectureTable.setItem(rowCount - 1, 1, QtGui.QTableWidgetItem('(' + str(self.dropoutRate.value()) + ')'))
                self.config['architecture'].append({'type': self.layerType.currentText(), 'units':self.dropoutRate.value()})
            else:
                self.architectureTable.setItem(rowCount - 1, 1, QtGui.QTableWidgetItem(str(self.hiddenUnits.value())))
                self.config['architecture'].append({'type': self.layerType.currentText(), 'units':self.hiddenUnits.value()})

        else:

            self.architectureTable.insertRow(self.architectureTable.currentRow() + 1)
            self.architectureTable.setItem(self.architectureTable.currentRow() + 1, 0, QtGui.QTableWidgetItem(self.layerType.currentText()))
            if self.layerType.currentText() == 'Dropout':
                self.architectureTable.setItem(self.architectureTable.currentRow() + 1, 1, QtGui.QTableWidgetItem('(' + str(self.dropoutRate.value()) + ')'))
                self.config['architecture'].insert(self.architectureTable.currentRow(), {'type': self.layerType.currentText(), 'units':self.dropoutRate.value()})
            else:
                self.architectureTable.setItem(self.architectureTable.currentRow() + 1, 1, QtGui.QTableWidgetItem(str(self.hiddenUnits.value())))
                self.config['architecture'].insert(self.architectureTable.currentRow(), {'type': self.layerType.currentText(), 'units':self.hiddenUnits.value()})



    def deleteRow(self):

        if self.architectureTable.currentRow() == -1 or self.architectureTable.currentRow() == 0 or self.architectureTable.currentRow() == self.architectureTable.rowCount() - 1 or self.architectureTable.rowCount() == 3:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("No se puede eliminar la fila")
            msg.setText("Seleccione o verifique la seleccion de la fila. Las capas de entrada y salida no pueden ser eliminadas, ademas la red debe tener como minimo una capa oculta.")      
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.exec_()
        else:
            self.architectureTable.removeRow(self.architectureTable.currentRow())
            del self.config['architecture'][self.architectureTable.currentRow()]


if __name__ == "__main__":

    import sys

    app = QtWidgets.QApplication(sys.argv)
    train = TrainWindow()

    train.show()
    
    sys.exit(app.exec_())