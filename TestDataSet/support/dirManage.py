import os

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)


def newDirectoryTest(*, principal="Principale", sub=None):
    """
    Crea nella directory corrente del progetto una nuova struttura a directory
    per il salvataggio delle info sul test effettuato
    :param principal: Directory principale da creare
    :param sub: sottodirectory da creare all'interno di quella principal
    """
    if sub is None:
        sub = ["Sub3", "Sub2"]

    os.makedirs(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{principal}")

    for dire in sub:
        os.makedirs(f"/Users/stefanobiddau/Desktop/Tesi/TestDataSet/images_output/{principal}/{dire}")

    logging.info("NUOVE DIRECTORIES PER IL TEST CREATE!")
