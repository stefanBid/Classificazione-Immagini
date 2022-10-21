# from grafics import printMNISTimage  # Se si desidera stampare un esempio di dati di MNIST
import os
import time

from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from support.grafics import printConfMatrix, printErroneusClassifications
from support.mnist import load_mnist

"""
********************************************************************************************
****    Questo script python testa il classificatore Alberi Decisionali                 ****
****    sul set di dati MNIST                                                           ****
****    Al termine dell'esecuzione di questo script avreo a disposizione:               **** 
****    1) La matrice di confusione sul set di train e di test di MNIST;                ****
****    2) Le immagini dei test falliti                                                 ****
****    3) Le informazioni relative ad ACCURANCY e LOG LOSS sul test e il tempo         ****
****       impiegato per addestrare il classificatore                                   ****
********************************************************************************************
"""


def over_under_fitting_controls(*, Y, Y_predicts):
    """
    Determina i valori MSE sul set di train e di test del target
    :param Y: Tupla contenete (set di train, set di test) del target
    :param Y_predicts: Tupla contenente  (predizione set di train, predizione set di test)
    :return: stringa contenete i valori di MSE per set di train e di test in modo da determinare eventuale ovr_f/und_f
    """
    mse_train = round(mean_squared_error(Y[0], Y_predicts[0]), 3)
    mse_test = round(mean_squared_error(Y[1], Y_predicts[1]), 3)
    return f"\t+ TRAIN\t\tMSE: {mse_train}\n\t+ TEST\t\tMSE: {mse_test}\n"


# Carico i dati all'interno dei vettori di train e di test

start = time.time()  # Tempo inizio processo
X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")
end = time.time()  # tempo fine processo

# Standaredizzo il dataset perche le immagini sono in scala da 1 a 256 noi li portiamo in scala da 0 a 1
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


k = 30  # Indica la profondità dell'albero

dotFile = open("Test_History.txt", "a")
if k == 0:  # Caso base
    dotFile.write(f"\n- Test del {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} MNIST con Decision Tree:\n")
    dotFile.write(f"\t+ Tempo caricamento dataset: {round((end - start), 3)} sec\n")
    dc = DecisionTreeClassifier(criterion="gini")
else:  # Devo settare il parametro della rete che riguarda i vicini
    dotFile.write(f"\n- Test del {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} MNIST con Decision Tree (prof: "
                  f"{k}):\n")
    dotFile.write(f"\t+ Tempo caricamento dataset: {round((end - start), 3)} sec\n")
    dc = DecisionTreeClassifier(max_depth=k)

# Addestro la rete sugli array data e target generati
start = time.time()  # Tempo inizio processo
dc.fit(X_train, Y_train)
Y_predict_test = dc.predict(X_test)
Y_proba_test = dc.predict_proba(X_test)
Y_predict_train = dc.predict(X_train)
Y_proba_train = dc.predict_proba(X_train)
end = time.time()  # tempo fine processo

# Calcolo le metriche di accuratezza sui set di train e di test
acc_train = round((accuracy_score(Y_train, Y_predict_train)*100), 2)
acc_test = round((accuracy_score(Y_test, Y_predict_test)*100), 2)
ll_train = round(log_loss(Y_train, Y_proba_train), 5)
ll_test = round(log_loss(Y_test, Y_proba_test), 5)

# Annoto i risultati ottenuti nel file che contiene lo storico dei risultati
dotFile.write(f"\t+ Tempo addestramento modello: {round((end - start), 3)} sec\n")
dotFile.write(f"\t+ TRAIN\t\tACCURANCY: {acc_train}%\tLOG LOSS: {ll_train}\n")
dotFile.write(f"\t+ TEST\t\tACCURANCY: {acc_test}%\tLOG LOSS: {ll_test}\n")
dotFile.write(over_under_fitting_controls(Y=(Y_train, Y_test), Y_predicts=(Y_predict_train, Y_predict_test)))
dotFile.close()

# Stampo la matrice di confusione per i set di train e di test e gli errori di classificazione
name = f"Test_MNIST_DC_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
name1 = "Confusion_Matrix"
name2 = "Erroneus_Classifications"
os.makedirs(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{name}/{name1}")
os.makedirs(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{name}/{name2}")
if k == 0:
    printConfMatrix(namDir=f"{name}/{name1}",
                    namImg="CM_train",
                    namGraf="train-set MNIST con Decision Tree",
                    target=Y_train,
                    predict=Y_predict_train)

    printConfMatrix(namDir=f"{name}/{name1}",
                    namImg="CM_test",
                    namGraf="test-set MNIST con Decision Tree",
                    target=Y_test,
                    predict=Y_predict_test)
else:
    printConfMatrix(namDir=f"{name}/{name1}",
                    namImg="CM_train",
                    namGraf=f"train-set MNIST con Decision Tree profondo {k}",
                    target=Y_train,
                    predict=Y_predict_train)

    printConfMatrix(namDir=f"{name}/{name1}",
                    namImg="CM_test",
                    namGraf=f"test-set MNIST con Decision Tree profondo {k}",
                    target=Y_test,
                    predict=Y_predict_test)

printErroneusClassifications(namDir=f"{name}/{name2}",
                             data=X_test,
                             target=Y_test,
                             predict=Y_predict_test)
