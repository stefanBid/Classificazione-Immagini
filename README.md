# Classificazione delle immagini  
Questo progetto è caratterizzato da due fasi:
1. [Test classificatori su dataset MNIST e F-MNIST](#test-classificatori-su-dataset-mnist-e-f-mnist)
2. [Test Reti Neurali su dataset MNIST e F-MNIST](#test-reti-neurali-su-dataset-mnist-e-f-mnist)

### Struttura del progetto
Il progetto nella repository prevede la seguente oranizzazione:
* __TestDataSet__
  * images_output
  * MNIST
  * F_MNIST
  * *__support__*
    * mnist.py
    * f_mnist.py
    * grafics.py
  * dataMNIST_KNN.py
  * dataMNIST_decisionTree.py
  * dataMNIST_randomForest.py
  * dataMNIST_SVM.py
  * Test_History.txt
  
Nella seguente tabella viene mostrato il contenuto siogni cartella o file  

| Nome elemento             | Tipo          | Contenuto                                                                                                     |
|---------------------------|---------------|---------------------------------------------------------------------------------------------------------------|
| TestDataSet               | Directory     | Directory principale del progetto contiene tutte le dir/file sottostanti                                      |
| images_output             | Directory     | Directory contenente i test grafici  generati automaticamente dagli script python                             |
| MNIST                     | Directory     | Directory contenete il dataset MNIST compresso                                                                |
| F_MNIST                   | Directory     | Directory contenente il dataset F-MNIST compresso                                                             |
| support                   | Directory     | Directory contenete script python con funzioni di supporto per visualizzazioni grafiche e caricamento dataset |
| mnist.py                  | File Python   | Script per caricare il dataset MNIST                                                                          |
| f_mnist.py                | File Python   | Script per caricare il dataset F-MNIST                                                                        |
| grafics.py                | File Python   | Script per visualizzare grafici relativi al modello di addestramento utilizzato                               |
| dataMNIST_KNN.py          | File Python   | Script per addestrare una K_NN su dataset MNIST                                                               |
| dataMNIST_decisionTree.py | File Python   | Script per addestrare un albero decisionale su dataset MNIST                                                  |
| dataMNIST_randomForest.py | File Python   | Script per addestrare una random forest su dataset MNIST                                                      |
| dataMNIST_SVM.py          | File Python   | Script per addestrare una SVM su dataset MNIST                                                                |
| Test_History.txt          | File di testo | Contiene il salvataggio di tutti i test effettuati                                                            |

La directory `images_output` conterrà una directory creata automticamente per ogni test effettuato e al suo interno avremo la matrice di confusione sul set di train e di test del dataset utilizzato, e le prime 100 immagini del set di test classificate erroneamente.

Se si vuole caricare il progetto manualmente tenere conto della gerarchia sopra mostrata altrimenti è presente nel repository il file `progetto.zip` che permette di caricare autmaticamete il progetto sul proprio IDE (io ho utilizzato [PyCharm](https://www.jetbrains.com/pycharm/)). 


# Test classificatori su dataset MNIST e F-MNIST
In questa sezione si è analizzato l'utilizzo di alcuni dei più famosi classificatori nel ML addestrati sui set di dati MNIST e F-MNIST e i loro risultati sono stati messi a confronto per dedurre quali dei sue dataset sia il più efficiente.
I classificatori utilizzati sono stati:
- K-Neighbors Classifier
- Alberi decisionali
- Foreste Casuali
- Macchina a vettori di supporto (SVM)

## MNIST

### Importare il dataset
Per rendere utilizzabile il dataset MNIST, questo è stato importato nel progetto da locale.
Come prima accennato i file zip contenenti il dataset sono presenti nella cartella `MNIST` e per importali è stata utilizzata la funzione `load_mnist(path="percorso")` presente nello script `mnist.py`

```python
from support.mnist import load_mnist
X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")
```

### Alcune info sul dataset utilizzato
Di seguito viene mostrato il blocco di codice utilizzato per ottenere le info utili sul dataset che sono servite poi per effettuare dei test ottimali:

```python
# sul dataset

# dimensioni del set di tarin
print("Dimensions TRAIN SET: ",X_train.shape, "\n")
print("Dimensions TEST SET: ",X_test.shape, "\n")

# data types
print(X_train.info())
print(X_test.info())
```

__I risultati ottenuti sono i seguenti:__  
  
  
Dimensions TRAIN SET:  (60000, 784)  

Dimensions TEST SET:   (10000, 784)  

class:  ndarray  
shape:  (60000, 784)  
strides:  (784, 1)  
itemsize:  1  
aligned:  True  
contiguous:  True  
fortran:  False  
data pointer: 0x7fdcd9c00000  
byteorder:  little  
byteswap:  False  
type: uint8  
None  

class:  ndarray  
shape:  (10000, 784)  
strides:  (784, 1)  
itemsize:  1  
aligned:  True  
contiguous:  True  
fortran:  False  
data pointer: 0x7fdcdc8dd000  
byteorder:  little  
byteswap:  False  
type: uint8  
None  

### Testare l'accuratezza dei classificatori
Le API dei classificatori utilizzati per essere addestrati sui set di train e di test sono stati importati dalla libreria `sklearn`. Per maggiori informazioni sulla libreria [sklearn](https://scikit-learn.org/stable/modules/classes.html)
```python
# Classificatore KNN
from sklearn.neighbors import KNeighborsClassifier

# Alberi decisionali
from sklearn.tree import DecisionTreeClassifier

# Foreste Randomiche
from sklearn.ensemble import RandomForestClassifier

# Macchina a vettori di supporto SVM
from sklearn.svm import SVC
```
Altre API da `sklearn` utilizzate:
```python
# Per portare i sets sulla stessa scala (0-1)
from sklearn.preprocessing import MinMaxScaler

# Errore quadratico medio per il controllo dell'overfitting
from sklearn.metrics mean_squared_error

# Per calcolare l'accuratezza della stima
from sklearn.metrics import accuracy_score

# Per misurare il modo in cui sono statedeterminate le stime
from sklearn.metrics import log_loss
```

Per l'output grafico invece sono state realizzate le funzioni ex novo `printConfMatrix()`, `printErroneusClassifications()`  che fanno uso di altre funzioni della libreria `matplotlib`. Per maggiori informazioni sulla libreria [matplotlib](https://matplotlib.org/)
```python

from support.grafics import printConfMatrix, printErroneusClassifications
```
| **Funzione**                               | **Compito**                                                                |
|--------------------------------------------|----------------------------------------------------------------------------|
| **printConfMatrix(...param)**              | Genera un immagine .png della matrice di confusione su un set di dati      |
| **printErroneusClassifications(...param)** | Genera le immagini .png dei primi 100 elementi classificati in modo errato |

(Successive funzioni sono in fase di creazione per avere un quadro più chiaro e dettagliato del modello testato)

### Esempio
*Qui di seguito viene riportato un esempio di ciò che si ottiene dall'esecuzione di uno degli script python. L'esempio preso in cosiderazione è un test effettuato il 20/10/2022 terminato alle ore 20:35*
L'esempio prevede l'esecuzione dello script `dataMNIST_SVM.py`  

> [NB]
> Gli script forniscono un implementazione generale del modello, tuttavia si possno modificare i parametri del modello al momento della sua creazione per ottenere risultati differenti.

Nel caso del nostro esempio:
```python
# Su una macchina a vettori di supporto si possono utilizzare diverse funzioni kernerl
kf = ["linear", "rbf", "sigmoid", "poly"]
i = 1  # Modifica il suo valore tra 0 e 3 per scegliere una funzione
svm = SVC(kernel=kf[i])
```

Al termine dell'addestramento del modello si otterrà una nuova directory in `images_output`  che avrà la seguente sintassi `TEST_MNIST_nomModello_dataTest` nel nostro caso:  

* images_output
  * ... altri TEST
  * TEST_MNIST_SVM_2022_10_20_20_35_38
    * Confusion_Matrix
       * CM_test.png
       * CM_train.png
    * Erroneus_Classifications 
       * ... 
       * Errore_32.png
       * ...

**** CM_test.png
![CM_test](https://i.ibb.co/Sx5Gw92/CM-test.png)
**** CM_train.png
![CM_train]()
**** Errore_32.png
![Error]()
## F-MNIST
!!! In fase di completamento !!!  


# Test Reti Neurali su dataset MNIST e F-MNIST
!!! In Fase di progettazione !!!
