import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
from matplotlib.legend_handler import HandlerLine2D

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)


def printConfMatrix(*, namDir, namImg, namGraf="Grafico", target, predict):
    """
    Salva la rappresentazione grafica della matrice di confusione
    :param namDir: Nome della cartella che conterrà le foto
    :param namImg: Nome della foto
    :param namGraf: Titolo che comparirà sopra il grafico
    :param target: target del modello
    :param predict: predizione del target
    :return: immagine .jpg della rappresentazione grafica della matrice di confusione
    """

    cm = confusion_matrix(target, predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Matrice di Confusione: {namGraf}")
    plt.ylabel("Classe corretta")
    plt.xlabel("Classe predetta")
    plt.savefig(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{namDir}/{namImg}",
                dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"MATRICE DI CONFUSIONE: {namGraf} CREATA!")


def printErroneusClassificationsMNIST(*, namDir, data, target, predict):
    """
    Fornisce le immagini classificate erroneamente nel dataset MNIST
    :param namDir: Nome della cartella che conterrà le foto
    :param data: Data di addestramento
    :param target: Valore atteso
    :param predict: Valore predetto
    :return: immagine .jpg delle immagini classificate erroneamente
    """
    # Posizione 0 -> Cifra 0 ... Posizione 9 -> Cifra 9
    labels = ['Cifra 0',
              'Cifra 1',
              'Cifra 2',
              'Cifra 3',
              'Cifra 4',
              'Cifra 5',
              'Cifra 6',
              'Cifra 7',
              'Cifra 8',
              'Cifra 9']
    y = 0
    for i in range(0, len(data)):
        if target[i] != predict[i]:
            y = y + 1
            if y == 100:
                break
            plt.title(f"{labels[target[i]]} classificata come {labels[predict[i]]}")
            plt.imshow(data[i].reshape([28, 28]), cmap="gray")
            plt.savefig(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{namDir}/Errore_{i}",
                        dpi=300, bbox_inches='tight')
            plt.close()
    logging.info("TUTTE LE IMMAGINI DEI TEST ERRATI SONO STATE CREATE!")


def printErroneusClassificationsFMNIST(*, namDir, data, target, predict):
    """
    Fornisce le immagini classificate erroneamente nel dataset FMNIST
    :param namDir: Nome della cartella che conterrà le foto
    :param data: Data di addestramento
    :param target: Valore atteso
    :param predict: Valore predetto
    :return: immagine .jpg delle immagini classificate erroneamente
    """
    # Posizione 0 -> T-shirt/top  ... Posizione 9 -> Stivaletto
    labels = ['T-shirt/top',
              'Pantalone',
              'Maglione',
              'Vestito',
              'Cappotto',
              'Sandalo',
              'Camicia',
              'Sneaker',
              'Borsa',
              'Stivaletto']
    y = 0
    for i in range(0, len(data)):
        if target[i] != predict[i]:
            y = y + 1
            if y == 100:
                break
            plt.title(f"{labels[target[i]]} classificato/a come {labels[predict[i]]}")
            plt.imshow(data[i].reshape([28, 28]), cmap="gray")
            plt.savefig(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{namDir}/Errore_{i}",
                        dpi=300, bbox_inches='tight')
            plt.close()
    logging.info("TUTTE LE IMMAGINI DEI TEST ERRATI SONO STATE CREATE!")


def printAccGraph(*, namDir, acc, rang, titGraf='Range', n_x_label='Range', stp=None ):
    """
    Fornisce la rappresentazione grafica delle accuratezze in base al range di valutazione
    :param stp: Quanti salti deve fare un valore all'altro sull'asse delle x
    :param titGraf: Titolo del grafo in forma: Accuratezza per titGraf
    :param n_x_label: Nome dell'x_label
    :param namDir: Nome della cartella che conterrà le foto
    :param acc: Lista di liste dei valori di accuratezza
    :param rang: Range di valutazione
    :return: immagine .jpg della rappresentazione grafica delle accuratezze in base al range di valutazione
    """
    if stp is None:
        stp = 1.0

    loc = np.arange(stp, rang + 1, step=stp)
    line1, = plt.plot(range(int(stp), rang+1, int(stp)), acc[1], 'b', label='Train accuracy')
    line2, = plt.plot(range(int(stp), rang+1, int(stp)), acc[0], 'r', label='Test accuracy')
    plt.xticks(loc)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.title(f"Accuratezza per {titGraf}")
    plt.ylabel('Accuratezza')
    plt.xlabel(n_x_label)
    plt.savefig(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{namDir}/Accurancy_Graph",
                dpi=300, bbox_inches='tight')
    plt.close()


def printAccGraphSVM(*, namDir, acc, rang, titGraf='Range', namImg, n_x_label='Range'):
    """
    Fornisce la rappresentazione grafica delle accuratezze in base al range di valutazione
    :param titGraf: Titolo del grafo in forma: Accuratezza per titGraf
    :param namImg: Nome della foto
    :param n_x_label: Nome dell'x_label
    :param namDir: Nome della cartella che conterrà le foto
    :param acc: Lista di liste dei valori di accuratezza
    :param rang: Range di valutazione
    :return: immagine .jpg della rappresentazione grafica delle accuratezze in base al range di valutazione
    """

    line1, = plt.plot(rang, acc[1], 'b', label='Train accuracy')
    line2, = plt.plot(rang, acc[0], 'r', label='Test accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.title(f"Accuratezza per {titGraf}")
    plt.ylabel('Accuratezza')
    plt.xlabel(n_x_label)
    plt.ylim([0.00, 1])
    plt.xscale('log')
    plt.savefig(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{namDir}/Accurancy_Graph_{namImg}",
                dpi=300, bbox_inches='tight')
    plt.close()
