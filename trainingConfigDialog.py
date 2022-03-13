# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'trainingConfigDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_trainingConfigDialog(object):
    def setupUi(self, trainingConfigDialog):
        trainingConfigDialog.setObjectName("trainingConfigDialog")
        trainingConfigDialog.resize(624, 480)
        trainingConfigDialog.setMinimumSize(QtCore.QSize(624, 480))
        trainingConfigDialog.setMaximumSize(QtCore.QSize(624, 480))
        trainingConfigDialog.setLayoutDirection(QtCore.Qt.LeftToRight)
        trainingConfigDialog.setStyleSheet("#trainingConfigDialog {\n"
"    background: white;\n"
"}\n"
"\n"
"QPushButton {\n"
"    font-size: 16pt;\n"
"    border: 1px solid black;\n"
"    border-radius: 10px;\n"
"    background: whitesmoke\n"
"}\n"
"\n"
"#saveConfig:hover {\n"
"    color: white;\n"
"    background: green\n"
"}\n"
"\n"
"#cancelConfig:hover {\n"
"    color: white;\n"
"    background: red\n"
"}\n"
"\n"
"#trainingTab, #architectureTab {    \n"
"    background-color: rgb(220, 220, 220);\n"
"}\n"
"\n"
"#architectureTable {\n"
"    border: none;\n"
"}\n"
"\n"
"#architectureTable QHeaderView:section {\n"
"    background: whitesmoke;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        trainingConfigDialog.setModal(True)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(trainingConfigDialog)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(trainingConfigDialog)
        self.tabWidget.setMinimumSize(QtCore.QSize(600, 400))
        self.tabWidget.setMaximumSize(QtCore.QSize(600, 400))
        self.tabWidget.setObjectName("tabWidget")
        self.architectureTab = QtWidgets.QWidget()
        self.architectureTab.setObjectName("architectureTab")
        self.formLayout_4 = QtWidgets.QFormLayout(self.architectureTab)
        self.formLayout_4.setObjectName("formLayout_4")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_11 = QtWidgets.QLabel(self.architectureTab)
        self.label_11.setMinimumSize(QtCore.QSize(60, 0))
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.architectureTab)
        self.label_9.setMinimumSize(QtCore.QSize(60, 0))
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.architectureTab)
        self.label_10.setMinimumSize(QtCore.QSize(60, 0))
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 2, 1, 1)
        self.timeSteps = QtWidgets.QSpinBox(self.architectureTab)
        self.timeSteps.setMinimum(1)
        self.timeSteps.setMaximum(10000)
        self.timeSteps.setProperty("value", 200)
        self.timeSteps.setObjectName("timeSteps")
        self.gridLayout_2.addWidget(self.timeSteps, 1, 0, 1, 1)
        self.channels = QtWidgets.QSpinBox(self.architectureTab)
        self.channels.setMinimum(1)
        self.channels.setProperty("value", 4)
        self.channels.setObjectName("channels")
        self.gridLayout_2.addWidget(self.channels, 1, 1, 1, 1)
        self.classes = QtWidgets.QSpinBox(self.architectureTab)
        self.classes.setMinimumSize(QtCore.QSize(60, 0))
        self.classes.setMinimum(2)
        self.classes.setProperty("value", 4)
        self.classes.setObjectName("classes")
        self.gridLayout_2.addWidget(self.classes, 1, 2, 1, 1)
        self.formLayout_3.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.gridLayout_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(self.architectureTab)
        self.label_6.setMinimumSize(QtCore.QSize(80, 0))
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.dropoutRate = QtWidgets.QDoubleSpinBox(self.architectureTab)
        self.dropoutRate.setMinimumSize(QtCore.QSize(60, 0))
        self.dropoutRate.setMaximumSize(QtCore.QSize(60, 16777215))
        self.dropoutRate.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.dropoutRate.setDecimals(1)
        self.dropoutRate.setMaximum(1.0)
        self.dropoutRate.setProperty("value", 0.5)
        self.dropoutRate.setObjectName("dropoutRate")
        self.gridLayout.addWidget(self.dropoutRate, 1, 2, 1, 1)
        self.layerType = QtWidgets.QComboBox(self.architectureTab)
        self.layerType.setMinimumSize(QtCore.QSize(120, 22))
        self.layerType.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.layerType.setObjectName("layerType")
        self.layerType.addItem("")
        self.layerType.addItem("")
        self.layerType.addItem("")
        self.gridLayout.addWidget(self.layerType, 1, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.architectureTab)
        self.label_7.setMinimumSize(QtCore.QSize(60, 0))
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.architectureTab)
        self.label_8.setMinimumSize(QtCore.QSize(160, 0))
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 2, 1, 1)
        self.hiddenUnits = QtWidgets.QSpinBox(self.architectureTab)
        self.hiddenUnits.setMinimumSize(QtCore.QSize(60, 22))
        self.hiddenUnits.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.hiddenUnits.setMinimum(1)
        self.hiddenUnits.setMaximum(10000)
        self.hiddenUnits.setProperty("value", 8)
        self.hiddenUnits.setObjectName("hiddenUnits")
        self.gridLayout.addWidget(self.hiddenUnits, 1, 1, 1, 1)
        self.addLayer = QtWidgets.QPushButton(self.architectureTab)
        self.addLayer.setMinimumSize(QtCore.QSize(30, 30))
        self.addLayer.setMaximumSize(QtCore.QSize(30, 30))
        self.addLayer.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.addLayer.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/plus.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.addLayer.setIcon(icon)
        self.addLayer.setObjectName("addLayer")
        self.gridLayout.addWidget(self.addLayer, 1, 3, 1, 1)
        self.removeLayer = QtWidgets.QPushButton(self.architectureTab)
        self.removeLayer.setMinimumSize(QtCore.QSize(30, 30))
        self.removeLayer.setMaximumSize(QtCore.QSize(30, 30))
        self.removeLayer.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.removeLayer.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/minus.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.removeLayer.setIcon(icon1)
        self.removeLayer.setObjectName("removeLayer")
        self.gridLayout.addWidget(self.removeLayer, 1, 4, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.architectureTable = QtWidgets.QTableWidget(self.architectureTab)
        self.architectureTable.setMinimumSize(QtCore.QSize(460, 240))
        self.architectureTable.setTabletTracking(False)
        self.architectureTable.setAutoFillBackground(False)
        self.architectureTable.setStyleSheet("")
        self.architectureTable.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.architectureTable.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.architectureTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.architectureTable.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.architectureTable.setTextElideMode(QtCore.Qt.ElideLeft)
        self.architectureTable.setShowGrid(True)
        self.architectureTable.setGridStyle(QtCore.Qt.DashLine)
        self.architectureTable.setWordWrap(True)
        self.architectureTable.setObjectName("architectureTable")
        self.architectureTable.setColumnCount(2)
        self.architectureTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.architectureTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.architectureTable.setHorizontalHeaderItem(1, item)
        self.architectureTable.horizontalHeader().setVisible(True)
        self.architectureTable.horizontalHeader().setCascadingSectionResizes(False)
        self.architectureTable.horizontalHeader().setDefaultSectionSize(125)
        self.architectureTable.horizontalHeader().setMinimumSectionSize(39)
        self.architectureTable.horizontalHeader().setSortIndicatorShown(False)
        self.architectureTable.horizontalHeader().setStretchLastSection(True)
        self.architectureTable.verticalHeader().setCascadingSectionResizes(False)
        self.architectureTable.verticalHeader().setSortIndicatorShown(False)
        self.architectureTable.verticalHeader().setStretchLastSection(False)
        self.verticalLayout_4.addWidget(self.architectureTable)
        self.formLayout_3.setLayout(1, QtWidgets.QFormLayout.LabelRole, self.verticalLayout_4)
        self.formLayout_4.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.formLayout_3)
        self.tabWidget.addTab(self.architectureTab, "")
        self.trainingTab = QtWidgets.QWidget()
        self.trainingTab.setObjectName("trainingTab")
        self.formLayout_2 = QtWidgets.QFormLayout(self.trainingTab)
        self.formLayout_2.setObjectName("formLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pretrainedModelCheck = QtWidgets.QCheckBox(self.trainingTab)
        self.pretrainedModelCheck.setMinimumSize(QtCore.QSize(220, 0))
        self.pretrainedModelCheck.setObjectName("pretrainedModelCheck")
        self.verticalLayout_3.addWidget(self.pretrainedModelCheck)
        self.label = QtWidgets.QLabel(self.trainingTab)
        self.label.setMinimumSize(QtCore.QSize(100, 0))
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pretrainedModelPath = QtWidgets.QLineEdit(self.trainingTab)
        self.pretrainedModelPath.setMinimumSize(QtCore.QSize(500, 0))
        self.pretrainedModelPath.setObjectName("pretrainedModelPath")
        self.horizontalLayout_4.addWidget(self.pretrainedModelPath)
        self.loadPretrainedModel = QtWidgets.QToolButton(self.trainingTab)
        self.loadPretrainedModel.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.loadPretrainedModel.setObjectName("loadPretrainedModel")
        self.horizontalLayout_4.addWidget(self.loadPretrainedModel)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.label_2 = QtWidgets.QLabel(self.trainingTab)
        self.label_2.setMinimumSize(QtCore.QSize(100, 0))
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.dataPath = QtWidgets.QLineEdit(self.trainingTab)
        self.dataPath.setMinimumSize(QtCore.QSize(500, 0))
        self.dataPath.setObjectName("dataPath")
        self.horizontalLayout_3.addWidget(self.dataPath)
        self.loadDataPath = QtWidgets.QToolButton(self.trainingTab)
        self.loadDataPath.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.loadDataPath.setObjectName("loadDataPath")
        self.horizontalLayout_3.addWidget(self.loadDataPath)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.label_3 = QtWidgets.QLabel(self.trainingTab)
        self.label_3.setMinimumSize(QtCore.QSize(100, 0))
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.modelPath = QtWidgets.QLineEdit(self.trainingTab)
        self.modelPath.setMinimumSize(QtCore.QSize(500, 0))
        self.modelPath.setObjectName("modelPath")
        self.horizontalLayout_2.addWidget(self.modelPath)
        self.saveModelPath = QtWidgets.QToolButton(self.trainingTab)
        self.saveModelPath.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.saveModelPath.setObjectName("saveModelPath")
        self.horizontalLayout_2.addWidget(self.saveModelPath)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.verticalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.trainingTab)
        self.label_4.setMinimumSize(QtCore.QSize(70, 0))
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.epochNumber = QtWidgets.QSpinBox(self.trainingTab)
        self.epochNumber.setMinimumSize(QtCore.QSize(70, 0))
        self.epochNumber.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.epochNumber.setMaximum(10000)
        self.epochNumber.setProperty("value", 100)
        self.epochNumber.setObjectName("epochNumber")
        self.verticalLayout_2.addWidget(self.epochNumber)
        self.label_5 = QtWidgets.QLabel(self.trainingTab)
        self.label_5.setMinimumSize(QtCore.QSize(70, 0))
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.batchSize = QtWidgets.QSpinBox(self.trainingTab)
        self.batchSize.setMinimumSize(QtCore.QSize(70, 0))
        self.batchSize.setProperty("value", 32)
        self.batchSize.setObjectName("batchSize")
        self.verticalLayout_2.addWidget(self.batchSize)
        self.formLayout.setLayout(1, QtWidgets.QFormLayout.LabelRole, self.verticalLayout_2)
        self.formLayout_2.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.formLayout)
        self.tabWidget.addTab(self.trainingTab, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.saveConfig = QtWidgets.QPushButton(trainingConfigDialog)
        self.saveConfig.setMinimumSize(QtCore.QSize(140, 40))
        self.saveConfig.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.saveConfig.setObjectName("saveConfig")
        self.horizontalLayout.addWidget(self.saveConfig)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.cancelConfig = QtWidgets.QPushButton(trainingConfigDialog)
        self.cancelConfig.setMinimumSize(QtCore.QSize(140, 40))
        self.cancelConfig.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cancelConfig.setObjectName("cancelConfig")
        self.horizontalLayout.addWidget(self.cancelConfig)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_5.addLayout(self.verticalLayout)

        self.retranslateUi(trainingConfigDialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(trainingConfigDialog)

    def retranslateUi(self, trainingConfigDialog):
        _translate = QtCore.QCoreApplication.translate
        trainingConfigDialog.setWindowTitle(_translate("trainingConfigDialog", "Dialog"))
        self.label_11.setText(_translate("trainingConfigDialog", "Muestras"))
        self.label_9.setText(_translate("trainingConfigDialog", "Canales"))
        self.label_10.setText(_translate("trainingConfigDialog", "Clases"))
        self.label_6.setText(_translate("trainingConfigDialog", "Tipo de capa"))
        self.layerType.setItemText(0, _translate("trainingConfigDialog", "Dense"))
        self.layerType.setItemText(1, _translate("trainingConfigDialog", "LSTM"))
        self.layerType.setItemText(2, _translate("trainingConfigDialog", "Dropout"))
        self.label_7.setText(_translate("trainingConfigDialog", "Unidades"))
        self.label_8.setText(_translate("trainingConfigDialog", "Tasa (Solo para Dropout)"))
        self.architectureTable.setSortingEnabled(False)
        item = self.architectureTable.horizontalHeaderItem(0)
        item.setText(_translate("trainingConfigDialog", "Tipo"))
        item = self.architectureTable.horizontalHeaderItem(1)
        item.setText(_translate("trainingConfigDialog", "Unidades"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.architectureTab), _translate("trainingConfigDialog", "Arquitectura"))
        self.pretrainedModelCheck.setText(_translate("trainingConfigDialog", "Entrenar modelo preentrenado"))
        self.label.setText(_translate("trainingConfigDialog", "Cargar modelo"))
        self.loadPretrainedModel.setText(_translate("trainingConfigDialog", "..."))
        self.label_2.setText(_translate("trainingConfigDialog", "Cargra dataset"))
        self.loadDataPath.setText(_translate("trainingConfigDialog", "..."))
        self.label_3.setText(_translate("trainingConfigDialog", "Guardar modelo"))
        self.saveModelPath.setText(_translate("trainingConfigDialog", "..."))
        self.label_4.setText(_translate("trainingConfigDialog", "Epochs"))
        self.label_5.setText(_translate("trainingConfigDialog", "Batch"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.trainingTab), _translate("trainingConfigDialog", "Entrenamiento"))
        self.saveConfig.setText(_translate("trainingConfigDialog", "Guardar"))
        self.cancelConfig.setText(_translate("trainingConfigDialog", "Cancelar"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    trainingConfigDialog = QtWidgets.QDialog()
    ui = Ui_trainingConfigDialog()
    ui.setupUi(trainingConfigDialog)
    trainingConfigDialog.show()
    sys.exit(app.exec_())
