import time

from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from support.graphs import *
from support.dsLoad import load_mnist, load_f_mnist
from support.writer import writeAppend
from support.dirManage import *

import logging

logging.basicConfig(format='%(asctime)s * %(message)s',
                    level=logging.INFO)


"""
****************************************************************************************************
****    Questo script python testa il classificatore Alberi Decisionali                         ****
****    sui set di dati MNIST/F-MNIST                                                           ****
****    Al termine dell'esecuzione di questo script si avrà a disposizione:                     **** 
****    1) La matrice di confusione sul set di train e di test di MNIST/F-MNIST;                ****
****    2) Le immagini dei primi 100 test falliti                                               ****
****    3) ACCURANCY e LOG LOSS sul test e il tempo impiegato per addestrare il classificatore  ****
****       impiegato per addestrare il classificatore                                           ****
****    4) Report di classificazione per ogni classe del dataset                                ****
****************************************************************************************************

****************************************************************
****               ! SCELTA DEL DATASET !                   ****
****                                                        ****
****   Modifica il parametro (choice) nel seguente modo:    ****
****      0 -> MNIST dataset                                ****
****      1 -> F-MNIST dataset                              ****   
****                                                        ****
****************************************************************
"""

choice = 0
datasetType = ['MNIST', 'F_MNIST']


'**** SET PARAMETRI & FUNZIONI UTILI ****'

# Directory principale per il test
prDir = f"DT/Test_{datasetType[choice]}_DT_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
# Sub directory per il test
subDir1 = "Confusion_Matrix"
subDir2 = "Erroneus_Classifications"

# Creazione delle directory per il test
newDirectoryTest(principal=prDir,
                 sub=[subDir1, subDir2])

# File dove salvare il test
file = "test_files/Test_History.txt"

# Parametri con cui è possibile addestrare la rete
k = 35  # profondità massima -> influisce sull'accuratezza
ind_c = 0  # Indice array c
ind_mf = 0  # Indice array mf
ind_msl = 1  # Indice array msl

c = ["gini", "entropy"]  # Criterion
mf = [None, 0.5]  # Max features
msl = [1, 25, 50]  # Min samples leaf


def over_under_fitting_controls(*, Y, Y_predicts):
    """
    Determina i valori MSE sul set di train e di test del target
    :param Y: Tupla contenete (set di train, set di test) del target
    :param Y_predicts: Tupla contenente (predizione set di train, predizione set di test)
    :return: stringa contenete i valori di MSE per set di train e di test in modo da determinare eventuale ovr_f/und_f
    """
    mse_train = round(mean_squared_error(Y[0], Y_predicts[0]), 3)
    mse_test = round(mean_squared_error(Y[1], Y_predicts[1]), 3)
    return f"\t+ TRAIN\t\tMSE: {mse_train}\n\t+ TEST\t\tMSE: {mse_test}\n"


'**** ADDESTRAMENTO DEL MODELLO ****'

start = time.time()  # Tempo inizio processo
if choice == 0:
    # Caricamento dei dati di MNIST all'interno dei vettori di train e di test
    X_train, X_test, Y_train, Y_test = load_mnist(path="data/MNIST")
else:
    # Caricamento dei dati di F-MNIST all'interno dei vettori di train e di test
    X_train, Y_train = load_f_mnist(path='data/FMNIST', kind='train')
    X_test, Y_test = load_f_mnist(path='data/FMNIST', kind='t10k')

end = time.time()  # tempo fine processo


# Standardizzazione del dataset per le immagini (scala da 1 a 256 -> scala da 0 a 1)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


writeAppend(filename=file,
            text=f"\n- Test del {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {datasetType[choice]} "
                 f"con Decision Tree (profondità: {k}, c: {c[ind_c]}, mf: {mf[ind_mf]}, msl: {msl[ind_msl]}):\n")
writeAppend(filename=file,
            text=f"\t+ Tempo caricamento dataset: {round((end - start), 3)} sec")


# Creazione del modello
dt = DecisionTreeClassifier(criterion=c[ind_c],
                            max_depth=k,
                            max_features=mf[ind_mf],
                            min_samples_leaf=msl[ind_msl],
                            random_state=0)


logging.info("INIZIO ADDESTRAMENTO DEL MODELLO!")
start = time.time()  # Tempo inizio processo

# Addestramento della rete sugli array data e target generati
dt.fit(X_train, Y_train)
Y_predict_test = dt.predict(X_test)
Y_proba_test = dt.predict_proba(X_test)
Y_predict_train = dt.predict(X_train)
Y_proba_train = dt.predict_proba(X_train)

end = time.time()  # tempo fine processo
logging.info("FINE ADDESTRAMENTO DEL MODELLO!")


# Calcolo delle metriche di accuratezza sui set di train e di test
acc_train = round((accuracy_score(Y_train, Y_predict_train) * 100), 2)
acc_test = round((accuracy_score(Y_test, Y_predict_test) * 100), 2)
ll_train = round(log_loss(Y_train, Y_proba_train), 5)
ll_test = round(log_loss(Y_test, Y_proba_test), 5)


# Scrittura dei risultati ottenuti nel file
writeAppend(filename=file,
            text=f"\t+ Tempo addestramento modello: {round((end - start), 3)} sec")
writeAppend(filename=file,
            text=f"\t+ TRAIN\t\tACCURANCY: {acc_train}%\tLOG LOSS: {ll_train}")
writeAppend(filename=file,
            text=f"\t+ TEST\t\tACCURANCY: {acc_test}%\tLOG LOSS: {ll_test}")
writeAppend(filename=file,
            text=over_under_fitting_controls(Y=(Y_train, Y_test), Y_predicts=(Y_predict_train, Y_predict_test)))

writeAppend(filename=file,
            text=f"\t+ Report di classificazione\n\n{classification_report(Y_test, Y_predict_test)}")


# Creazione immagine della matrice di confusione per i set di train e di test e gli errori di classificazione
printConfMatrix(namDir=f"{prDir}/{subDir1}",
                namImg="CM_train",
                namGraf=f"train-set {datasetType[choice]} con Decision Tree "
                        f"(p: {k}, c: {c[ind_c]}, mf: {mf[ind_mf]}, msl: {msl[ind_msl]})",
                target=Y_train,
                predict=Y_predict_train)

printConfMatrix(namDir=f"{prDir}/{subDir1}",
                namImg="CM_test",
                namGraf=f"test-set {datasetType[choice]} con Decision Tree "
                        f"(p: {k}, c: {c[ind_c]}, mf: {mf[ind_mf]}, msl: {msl[ind_msl]})",
                target=Y_test,
                predict=Y_predict_test)

# Creazione delle immagini dei test classificati erroneamente
if choice == 0:
    printErroneusClassificationsMNIST(namDir=f"{prDir}/{subDir2}",
                                      data=X_test,
                                      target=Y_test,
                                      predict=Y_predict_test)
else:
    printErroneusClassificationsFMNIST(namDir=f"{prDir}/{subDir2}",
                                       data=X_test,
                                       target=Y_test,
                                       predict=Y_predict_test)
