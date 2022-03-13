##########################################################################################################################
##  Programa para generar dataset artificial
##########################################################################################################################

import numpy as np
import pickle
from random import randrange, shuffle


n_classes = 4
time_steps = 200

X = []
Y = []
a = np.zeros((time_steps,n_classes))

##  Clases
for i in range(n_classes):

    ##  Repeticiones de cada clase
    for j in range(100):

        ##  Numero de pasos o tiempos t (Tx)
        for k in range(time_steps):

            if (i == 0):

                a[k,0] = randrange(-10,11)
                a[k,0] = randrange(-50,49)
                a[k,0] = randrange(-100,101)
                a[k,0] = randrange(-254,254)

            elif (i == 1):

                a[k,0] = randrange(-100,101)
                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-50,49)
                a[k,0] = randrange(-10,11)

            elif (i == 2):

                a[k,0] = randrange(-100,101)
                a[k,0] = randrange(-10,11)
                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-50,49)

            elif (i == 3):

                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-10,11)
                a[k,0] = randrange(-100,101)
                a[k,0] = randrange(-50,49)

            elif (i == 4):

                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-50,49)
                a[k,0] = randrange(-10,11)
                a[k,0] = randrange(-100,101)

            elif (i == 5):

                a[k,0] = randrange(-10,11)
                a[k,0] = randrange(-50,49)
                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-254,254)

            elif (i == 6):

                a[k,0] = randrange(-100,101)
                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-50,49)
                a[k,0] = randrange(-10,11)

            elif (i == 7):

                a[k,0] = randrange(-254,254)
                a[k,0] = randrange(-100,101)
                a[k,0] = randrange(-50,49)
                a[k,0] = randrange(-254,254)

        X.append(a)
        Y.append(i)
        a = np.zeros((time_steps,n_classes))

##  Convierte X y Y en un tuple para guardarlos juntos
c = list(zip(X, Y))

##  Reordena el dataset aleatoriamente
shuffle(c)

##  Guarda el dataset en un archivo pkl
with open('dataset.pkl', 'wb') as f:
    pickle.dump(c, f)


