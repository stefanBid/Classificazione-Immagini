import time

from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from support.dirManage import newDirectoryTest
from support.graphs import *
from support.dsLoad import load_mnist, load_f_mnist
from support.writer import writeAppend

"""
****************************************************************************************************
****    Questo script python testa il classificatore SVM sui set di dati MNIST/F-MNIST          ****
****    Al termine dell'esecuzione di questo script si avrà a disposizione:                     **** 
****    1) La matrice di confusione sul set di train e di test di MNIST/F-MNIST;                ****
****    2) Le immagini dei primi 100 test falliti                                               ****
****    3) ACCURANCY sul test e il tempo impiegato per addestrare il classificatore             ****
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

choice = 1
datasetType = ['MNIST', 'F_MNIST']


'**** SET PARAMETRI & FUNZIONI UTILI ****'

# Directory principale per il test
prDir = f"SVM/Test_{datasetType[choice]}_SVM_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
# Sub directory per il test
subDir1 = "Confusion_Matrix"
subDir2 = "Erroneus_Classifications"

# Creazione delle directory per il test
newDirectoryTest(principal=prDir,
                 sub=[subDir1, subDir2])

# Nome del file dove salvare il test
file = "test_files/Test_History.txt"


# Parametri per cui è possibile addestrare il classificatore
ind_k = 3  # Indice array k
ind_c = 0  # Indice array c
ind_g = 0  # Indice array g ind_g = 3 se si vuole ignorare il parametro

k = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']  # kernel
c = [5, 10]  # Regolarizzazione parametri
g = [1e-2, 1e-3, 1e-4, "scale"]  # Coefficiente del kernel
# ignorare questo parametro se non si usano kernel poly, rbf, sigmoid


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

start = time.time()  # tempo inizio processo

if choice == 0:
    # Caricamento dei dati di MNIST all'interno dei vettori di train e di test
    X_train, X_test, Y_train, Y_test = load_mnist(path="data/MNIST")
else:
    # Caricamento dei dati di F-MNIST all'interno dei vettori di train e di test
    X_train, Y_train = load_f_mnist(path='data/FMNIST', kind='train')
    X_test, Y_test = load_f_mnist(path='data/FMNIST', kind='t10k')

end = time.time()  # tempo fine processo


# Standardizzazione del dataset (scala da 1 a 256 -> scala da 0 a 1)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


writeAppend(filename=file,
            text=f"\n- Test del {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {datasetType[choice]} con SVM "
                 f"(k: {k[ind_k]}, c: {c[ind_c]}, g: {g[ind_g]}):\n")
writeAppend(filename=file,
            text=f"\t+ Tempo caricamento dataset: {round((end - start), 3)} sec")


# Creazione del modello
svc = SVC(kernel=k[ind_k], C=c[ind_c], gamma=g[ind_g])

logging.info("INIZIO ADDESTRAMENTO DEL MODELLO!")
start = time.time()  # Tempo inizio processo

# Addestramento della rete sugli array data e target generati
svc.fit(X_train[:6000], Y_train[:6000])
Y_predict_test = svc.predict(X_test[:1000])
Y_predict_train = svc.predict(X_train[:6000])

end = time.time()  # tempo fine processo
logging.info("FINE ADDESTRAMENTO DEL MODELLO!")


# Calcolo delle metriche di accuratezza sui set di train e di test
acc_train = round((svc.score(X_train[:6000], Y_train[:6000])*100), 2)
acc_test = round((svc.score(X_test[:1000], Y_test[:1000])*100), 2)


# Scrittura nel file di testo dei risultati ottenuti
writeAppend(filename=file,
            text=f"\t+ Tempo addestramento modello: {int((end - start) / 60)} min")
writeAppend(filename=file,
            text=f"\t+ TRAIN\t\tACCURANCY: {acc_train}%")
writeAppend(filename=file,
            text=f"\t+ TEST\t\tACCURANCY: {acc_test}%")
writeAppend(filename=file,
            text=over_under_fitting_controls(Y=(Y_train[:6000], Y_test[:1000]), Y_predicts=(Y_predict_train, Y_predict_test)))

writeAppend(filename=file,
            text=f"\t+ Report di classificazione\n\n{classification_report(Y_test[:1000], Y_predict_test)}")

# Stampa della matrice di confusione per i set di train e di test e gli errori di classificazione

printConfMatrix(namDir=f"{prDir}/{subDir1}",
                namImg="CM_train",
                namGraf=f"Train-set {datasetType[choice]} con SVM (k: {k[ind_k]}, c: {c[ind_c]}, g: {g[ind_g]})",
                target=Y_train[:6000],
                predict=Y_predict_train)

printConfMatrix(namDir=f"{prDir}/{subDir1}",
                namImg="CM_test",
                namGraf=f"Test-set {datasetType[choice]} con SVM (k: {k[ind_k]}, c: {c[ind_c]}, g: {g[ind_g]})",
                target=Y_test[:1000],
                predict=Y_predict_test)

# Creazione delle immagini dei test classificati erroneamente
if choice == 0:
    printErroneusClassificationsMNIST(namDir=f"{prDir}/{subDir2}",
                                      data=X_test[:1000],
                                      target=Y_test[:1000],
                                      predict=Y_predict_test)
else:
    printErroneusClassificationsFMNIST(namDir=f"{prDir}/{subDir2}",
                                       data=X_test[:1000],
                                       target=Y_test[:1000],
                                       predict=Y_predict_test)






