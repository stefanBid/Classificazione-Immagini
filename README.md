# Classificazione delle immagini  
Questo progetto è caratterizzato da due fasi:
1. [Test classificatori su dataset MNIST e F-MNIST](#test-classificatori-su-dataset-mnist-e-f-mnist)
2. [Test Reti Neurali su dataset MNIST e F-MNIST](#test-reti-neurali-su-dataset-mnist-e-f-mnist)

### Struttura del progetto
Il progetto nella repository prevede la seguente oranizzazione:
* __TestDataSet__
  * images_output
  * test_files
  * *__support__*
    * ___init___.py
    * dirManage.py
    * dsLoad.py
    * graphs.py
    * write.py
  * decisionTree_classifier.py
  * decisionTree_classifier_HO.py
  * KNN_classifier.py
  * KNN_classifier_HO.py
  * randomForest_classifier.py
  * randomForest_classifier_HO.py
  * SVM_classifier.py
  * SVM_classifier_HO.py
  
  
Nella seguente tabella viene mostrato il contenuto siogni cartella o file  


| **Nome**                      | **Tipo**       | **Descrizione**                                                                                     |
|-------------------------------|----------------|-----------------------------------------------------------------------------------------------------|
| TestDataSet**                 | Direcrtory.    | Directory principale del progetto contiene tutte le dir/file sottostanti                            |
| images_output                 | Directory      | Directory contenente i test grafici  generati automaticamente dagli script python                   |
| test_files                    | Directory      | Directory contenente i file di testo nel quale vengono annotati i risultati dei test                |
| support                       | Package Python | Package contenente script python di supporto                                                        |
| decisionTree_classifier.py    | File Python    | Script per addestrare un albero decisionale su dataset MNIST/F-MNIST                                |
| decisionTree_classifier_HO.py | File Python    | Script per testare gli Iper-parametri dell' albero decisionale per ottenere la maggiore accuratezza |
| KNN_classifier.py             | File Python    | Script per addestrare una KNN su dataset MNIST/F-MNIST                                              |
| KNN_classifier_HO.py          | File Python    | Script per testare gli Iper-parametri della KNN per ottenere la maggiore accuratezza                |
| randomForest_classifier.py    | File Python    | Script per addestrare una foresta randomica su dataset MNIST/F-MNIST                                |
| randomForest_classifier_HO.py | File Python    | Script per testare gli Iper-parametri della foresta randomica per ottenere la maggiore accuratezza  |
| SVM_classifier.py             | File Python    | Script per addestrare una SVM su dataset MNIST/F-MNIST                                              |
| SVM_classifier_HO.py          | File Python    | Script per testare gli Iper-parametri della SVM per ottenere la maggiore accuratezza                |



La sub-directory `images_output` conterrà a suo interno le directory create automaticamente dagli script nel quale ci saranno i Grafici di accuratezza dei test effettuati sui vari classificatori.  

La sub-directory `test_files`invece conterrà a suo interno i file:
* `Test_History.txt` : Conterrà i risultati ottenuti dai test effettuati sui classificatori;
* `HO_History.txt` : Conterrà i risultati ottenutti dai test effettuati combinando i vari Iper-parametri sui classificatori; 


Se si vuole caricare il progetto manualmente tenere conto della gerarchia sopra mostrata altrimenti scaricare la repository in formato .zip che permette di caricare autmaticamete il progetto sul proprio IDE (io ho utilizzato [PyCharm](https://www.jetbrains.com/pycharm/)). 


# Test classificatori su dataset MNIST e F-MNIST
In questa sezione si è analizzato l'utilizzo di alcuni dei più famosi classificatori nel ML addestrati sui set di dati MNIST e F-MNIST e i loro risultati sono stati messi a confronto per dedurre quali dei due dataset sia il più efficiente, tenendo conto di un importantissimo aspetto ossia che il dataset MNIST sia più semplice di F-MNIST, per approfondire l'argomento visitare la repository di [zalandoresearch](https://github.com/zalandoresearch/fashion-mnist).  


I classificatori utilizzati sono stati:
- K-Neighbors Classifier
- Alberi decisionali
- Foreste Casuali
- Macchina a vettori di supporto (SVM)

## MNIST

### Importare il dataset
Per rendere utilizzabili i dataset MNIST e F-MNIEST, questi sono stati importati nel progetto da locale.
Come prima accennato i file zip contenenti il dataset sono presenti ripesttivamente nelle directtory `data/MNIST` e `data/FMNIST` e per importali sono state utilizzate le funzioni `load_mnist(path="percorso")` e `load_f_mnist_mnist(path="percorso", kind="train")`,  presente nello script `dsLoad.py`.

```python
from support.mnist import load_mnist, load_f_mnist

# Caricamento dei dati di MNIST all'interno dei vettori di train e di test
X_train, X_test, Y_train, Y_test = load_mnist(path="data/MNIST")

# Caricamento dei dati di F-MNIST all'interno dei vettori di train e di test
X_train, Y_train = load_f_mnist(path='data/FMNIST', kind='train')
X_test, Y_test = load_f_mnist(path='data/FMNIST', kind='t10k')
```

### Alcune info sui dataset MNIST/F-MNIST
Di seguito viene mostrato il blocco di codice utilizzato per ottenere le info utili sui dataset che sono servite poi per effettuare dei test ottimali:

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
Le API dei classificatori utilizzati per essere addestrati sui set di train e di test sono stati importati dalla libreria `sklearn`. Per maggiori informazioni sulla libreria [sklearn](https://scikit-learn.org/stable/modules/classes.html).
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

# Per mostrare la ripartizione dell'accuratezza tra le varie classi di predizione
from sklearn.metrics import classification_report

# Per creare il modello che effettua delle stime di predizione combinando vari Iper-parametri
from sklearn.model_selection import GridSearchCV
```

Per l'output grafico invece sono state realizzate delle funzioni ex novo che fanno uso di altre funzioni della libreria `matplotlib`. Per maggiori informazioni sulla libreria [matplotlib](https://matplotlib.org/).

```python

from support.grafics import printConfMatrix, printErroneusClassificationsMNIST, printErroneusClassificationsFMNIST, printAccGraph, printAccGraphSVM
```

| **Funzione**                                     | **Compito**                                                                           |
|--------------------------------------------------|---------------------------------------------------------------------------------------|
| **printConfMatrix(...param)**                    | Genera un immagine .png della matrice di confusione su un set di dati                 |
| **printErroneusClassificationsMNIST(...param)**  | Genera le immagini .png dei primi 100 elementi classificati in modo errato su MNIST   |
| **printErroneusClassificationsFMNIST(...param)** | Genera le immagini .png dei primi 100 elementi classificati in modo errato su F-MNIST |
| **printErroneusClassificationsFMNIST(...param)** | Genera le immagini .png dei primi 100 elementi classificati in modo errato su F-MNIST |
| **printAccGraph(...param)**                      | Genera un immagine .png del grafico di accuratezza tra train set e test set           |
| **printAccGraphSVM(...param)**                   | Genera un immagine .png del grafico di accuratezza tra train set e test set per SVM   |

Per annotare i risultati dei test nei due file di testo è stata usata una funzione ex novo che fa uso della funzione builtins per la scrittura/lettura dei file offerta da python.

```python

from support.writer import writeAppend 
```


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
       * Errore_844.png
       * ...

#### CM_test.png
![CM_test](https://i.ibb.co/Sx5Gw92/CM-test.png)
#### CM_train.png
![CM_train](https://i.ibb.co/KWqtGSc/CM-train.png)
#### Errore_844.png
![Error](https://i.ibb.co/KjQ1kX1/Errore-844.png)

Infine nel file di testo `Test_History.txt` sarà scritto il nuovo test effettuato con tutte le sue caratteristiche di seguito viene mostrato il formato

- Test del 2022-10-20 20:23:33 MNIST con SVM con kernel (linear)
	+ Tempo caricamento dataset: 0.036 sec
	+ Tempo addestramento modello: 7 min
	+ TRAIN		ACCURANCY: 97.08%
	+ TEST		ACCURANCY: 94.01%
	+ TRAIN		MSE: 0.477
	+ TEST		MSE: 1.019


# Test Reti Neurali su dataset MNIST e F-MNIST
!!! In Fase di progettazione !!!
