import time
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

from support.dirManage import *
from support.writer import writeAppend
from support.graphs import printAccGraphSVM
from support.dsLoad import load_mnist, load_f_mnist

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)

"""
****************************************************************************************************
****    Questo script python testa il classificatore SVM al variare degli Iper_parametri        ****
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

choice = 1
datasetType = ['MNIST', 'F_MNIST']


'**** SET PARAMETRI & FUNZIONI UTILI ****'

# Directory principale per il test
prDir = f"SVM/Hyperparameter_Tuning_{datasetType[choice]}_SVM_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
# Sub directory per il test
subDir1 = "Accurancy_Graphic"

# Creazione delle directory per l'ottimizzazione dei parametri
newDirectoryTest(principal=prDir,
                 sub=[subDir1])

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

logging.info(f"DATASET {datasetType[choice]} CARICATO!")

# Standardizzazione del dataset (scala da 1 a 256 -> scala da 0 a 1)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


# Dizionario degli Iper-parametri sul quale effettuare i test
k = ['poly', 'rbf', 'sigmoid']
grid_params = {'gamma': [1e-2, 1e-3, 1e-4],
               'C': [5, 10]}

writeAppend(filename=file,
            text=f"\nSVM su {datasetType[choice]}")

logging.info("INIZIO TEST IPER-PARAMETRI")

for k_el in k:
    logging.info(f"TEST CON KERNEL {k_el} ")

    gs = GridSearchCV(SVC(kernel=k_el),
                      grid_params,
                      verbose=1,
                      cv=4,
                      n_jobs=-1,
                      scoring="accuracy",
                      return_train_score=True)

    # Addestramento del modello che combinerà i vari parametri alla ricerca dei migliori risultati
    g_res = gs.fit(X_train[:6000], Y_train[:6000])

    # Settaggio delle opzioni di Pandas per visualizzare interamente una tabella con molte colonne
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Score_df sarà un DataFrame che conterrà per ogni riga i risultati ottenuti per ogni combinazione
    score_df = pd.DataFrame(g_res.cv_results_)

    writeAppend(filename=file,
                text=f"\n\tCon kernel: {k_el}")

    writeAppend(filename=file,
                text=f"\tIl tasso di accuratezza si aggirerà intorno al : {round(g_res.best_score_ * 100, 2)}%\n\n")

    writeAppend(filename=file,
                text=score_df.nlargest(5, "mean_test_score"))

    writeAppend(filename=file,
                text=f"\n\n\tMigliori Iper-parametri: {g_res.best_params_}")

    # Preparazione parametri dei grafici
    score_df['param_C'] = score_df['param_C'].astype('int')
    gamma_01 = score_df[score_df['param_gamma'] == 0.01]
    gamma_001 = score_df[score_df['param_gamma'] == 0.001]
    gamma_0001 = score_df[score_df['param_gamma'] == 0.0001]

    printAccGraphSVM(namDir=f"{prDir}/{subDir1}",
                     acc=[gamma_01["mean_test_score"], gamma_01["mean_train_score"]],
                     rang=gamma_01["param_C"],
                     titGraf=f"SVM con kernel {k_el} e gamma 0,01",
                     namImg=f"{k_el}_gamma01",
                     n_x_label='C'
                     )

    printAccGraphSVM(namDir=f"{prDir}/{subDir1}",
                     acc=[gamma_01["mean_test_score"], gamma_01["mean_train_score"]],
                     rang=gamma_01["param_C"],
                     titGraf=f"SVM con kernel {k_el} e gamma 0,001",
                     namImg=f"{k_el}_gamma001",
                     n_x_label='C'
                     )

    printAccGraphSVM(namDir=f"{prDir}/{subDir1}",
                     acc=[gamma_0001["mean_test_score"], gamma_0001["mean_train_score"]],
                     rang=gamma_0001["param_C"],
                     titGraf=f"SVM con kernel {k_el} e gamma 0,0001",
                     namImg=f"{k_el}_gamma0001",
                     n_x_label='C'
                     )
