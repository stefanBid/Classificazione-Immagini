# Classificazione delle immagini  
Questo progetto è caratterizzato da due fasi:
1. [Test classificatori su dataset MNIST e F-MNIST](#test-classificatori-su-dataset-mnist-e-f-mnist)
2. [Test Reti Neurali su dataset MNIST e F-MNIST](#test-reti-neurali-su-dataset-mnist-e-f-mnist)

### Struttura del progetto
Il progetto nella repository prevede la seguente oranizzazione:
- TestDataSet
 -  images_output
 -  MNIST
 -  F-MNIST
 -  support
  - mnist.py
  - f_mnist.py
  - grafics.py
 -  dataMNIST_KNN.py
 -  dataMNIST_decisionTree.py
 -  dataMNIST_randomForest.py
 -  dataMNIST_SVM.py
 -  Test_History.txt
  
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




# Test classificatori su dataset MNIST e F-MNIST
In questa sezione si è analizzato l'utilizzo di alcuni dei più famosi classificatori nel ML addestrati sui set di dati MNIST e F-MNIST e i loro risultati sono stati messi a confronto per dedurre quali dei sue dataset sia il più efficiente.
I classificatori utilizzati sono stati:
- K-Neighbors Classifier
- Alberi decisionali
- Foreste Casuali
- Macchina a vettori di supporto (SVM)


# Test Reti Neurali su dataset MNIST e F-MNIST
