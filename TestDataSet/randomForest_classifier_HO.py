import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from support.dirManage import *
from support.writer import writeAppend
from support.graphs import printAccGraph
from support.dsLoad import load_mnist, load_f_mnist

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)

"""
****************************************************************************************************
****    Questo script python testa il classificatore Foreste Randomiche                         ****
****    al variare degli Iper_parametri                                                         ****
****    Al termine dell'esecuzione di questo script si avrà a disposizione:                     **** 
****    1) La predizione dei migliori iper-parametri per addestrare la rete                     ****
****    2) Grafico di come varia l'accuratezza del Test e del Train in base agli Iper-parametri ****
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
prDir = f"RF/Hyperparameter_Tuning_{datasetType[choice]}_RF_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
# Sub directory per il test
subDir1 = "Accurancy_Graphic_pro"
subDir2 = "Accurancy_Graphic_pop"

# Creazione delle directory per l'ottimizzazione dei parametri
newDirectoryTest(principal=prDir,
                 sub=[subDir1, subDir2])


# Profondità max dell'albero decisionale
k = 21

# Popolazione max della foresta
p = 200


# Titolo del documento in cui salvare le informazioni
file = "test_files/HO_History.txt"


'**** ESECUZIONE DEI TEST ****'

if choice == 0:
    # Caricamento dei dati di MNIST all'interno dei vettori di train e di test
    X_train, X_test, Y_train, Y_test = load_mnist(path="data/MNIST")
else:
    # Caricamento dei dati di F-MNIST all'interno dei vettori di train e di test
    X_train, Y_train = load_f_mnist(path='data/FMNIST', kind='train')
    X_test, Y_test = load_f_mnist(path='data/FMNIST', kind='t10k')


# Standardizzazione del dataset per le immagini (scala da 1 a 256 -> scala da 0 a 1)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


# Calcolo accuratezza in base alla profondità
train_acc = np.zeros(k)
test_acc = np.zeros(k)


logging.info("INIZIO SEQUENZA DI ADDESTRAMENTO PER PROFONDITA:")

for i in range(1, k+1):
    dt = RandomForestClassifier(max_depth=i, random_state=0).fit(X_train, Y_train)
    Y_predict_test = dt.predict(X_test)
    Y_predict_train = dt.predict(X_train)
    test_acc[i - 1] = round((accuracy_score(Y_test, Y_predict_test) * 100), 2)
    train_acc[i - 1] = round((accuracy_score(Y_train, Y_predict_train) * 100), 2)
    logging.info(f"FINE ADDESTRAMENTO DEL MODELLO CON PROFONDITA {i}")

logging.info("FINE SEQUENZA DI ADDESTRAMENTO")


# Creazione del grafico
printAccGraph(namDir=f"{prDir}/{subDir1}",
              acc=[test_acc, train_acc],
              rang=k,
              titGraf="massima profondità",
              n_x_label='Massima profondità')


# Calcolo accuratezza in base alla popolazione
train_acc1 = np.zeros(int(p/20))
test_acc1 = np.zeros(int(p/20))

logging.info("INIZIO SEQUENZA DI ADDESTRAMENTO PER POPOLAZIONE DI ALBERI:")
z = 1
for j in range(20, p+1, 20):
    dt = RandomForestClassifier(n_estimators=j, random_state=0).fit(X_train, Y_train)
    Y_predict_test = dt.predict(X_test)
    Y_predict_train = dt.predict(X_train)
    test_acc1[z - 1] = round((accuracy_score(Y_test, Y_predict_test) * 100), 2)
    train_acc1[z - 1] = round((accuracy_score(Y_train, Y_predict_train) * 100), 2)
    z = z + 1
    logging.info(f"FINE ADDESTRAMENTO DEL MODELLO CON POPOLAZIONE {j}")

logging.info("FINE SEQUENZA DI ADDESTRAMENTO")


# Creazione del grafico
printAccGraph(namDir=f"{prDir}/{subDir2}",
              acc=[test_acc1, train_acc1],
              rang=p,
              stp=20.0,
              titGraf="popolazione alberi",
              n_x_label='Popolazione Alberi')


# Dizionario degli Iper-parametri sul quale effettuare i test
grid_params = {'n_estimators': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
               'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 21],
               'min_samples_split': [2, 5, 10],
               'criterion': ["gini", "entropy"]}

gs = GridSearchCV(RandomForestClassifier(random_state=0),
                  grid_params,
                  verbose=1,
                  cv=4,
                  n_jobs=-1,
                  scoring="accuracy",
                  return_train_score=True)

# Addestramento del modello che combinerà i vari parametri alla ricerca dei migliori risultati
g_res = gs.fit(X_train, Y_train)

# Settaggio delle opzioni di Pandas per visualizzare interamente una tabella con molte colonne
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Score_df sarà un DataFrame che conterrà per ogni riga i risultati ottenuti per ogni combinazione
score_df = pd.DataFrame(g_res.cv_results_)


writeAppend(filename=file,
            text=f"\nRF su {datasetType[choice]}")

writeAppend(filename=file,
            text=f"\tIl tasso di accuratezza si aggirerà attorno al : {round(g_res.best_score_ * 100, 2)}%\n\n")

writeAppend(filename=file,
            text=score_df.nlargest(10, "mean_test_score", keep='last'))

writeAppend(filename=file,
            text=f"\n\n\tMigliori Iper-parametri: {g_res.best_params_}")
