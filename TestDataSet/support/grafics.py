import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


def printErroneusClassifications(*, namDir, data, target, predict):
    """
    Fornisce le immagini classificate erroneamente
    :param namDir: Nome della cartella che conterrà le foto
    :param data: Data di addestramento
    :param target: Valore atteso
    :param predict: Valore predetto
    :return: immagine .jpg delle immagini classificate erroneamente
    """
    y = 0
    for i in range(0, len(data)):
        if target[i] != predict[i]:
            y = y + 1
            if y == 100:
                break
            plt.title(f"Cifra {target[i]} classificata come {predict[i]}")
            plt.imshow(data[i].reshape([28, 28]), cmap="gray")
            plt.savefig(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{namDir}/Errore_{i}",
                        dpi=300, bbox_inches='tight')
            plt.close()

