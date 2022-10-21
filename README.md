# Classificazione delle immagini  
Questo progetto è caratterizzato da due fasi:
1. [Test classificatori su dataset MNIST e F-MNIST](#test-classificatori-su-dataset-mnist-e-f-mnist)
2. [Test Reti Neurali su dataset MNIST e F-MNIST](#test-reti-neurali-su-dataset-mnist-e-f-mnist)

### Struttura del progetto
Il progetto nella repository prevede la seguente oranizzazione:
- TestDataSet
  * images_output
  * MNIST
  * F-MNIST
  * support
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
| F-MNIST                   | Directory     | Directory contenente il dataset F-MNIST compresso                                                             |
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
print("Dimensions TEST SET: ",X_train.shape, "\n")

# data types
print(X_train.info())
print(X_test.info())
```

I risultati ottenuti sono i seguenti:  
Dimensions TRAIN SET:  (60000, 784)  

Dimensions TEST SET:  (60000, 784)  

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


# Test Reti Neurali su dataset MNIST e F-MNIST
