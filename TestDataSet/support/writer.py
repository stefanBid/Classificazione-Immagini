import os
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)


def writeAppend(*, filename="file.txt", text="Information"):
    """
    Scrive in append informazioni in un file di testo
    :param filename: Nome del file nel quale si vuole scrivere
    :param text: Testo da scrivere nel file
    """
    dotFile = open(filename, "a")
    dotFile.write(f"{text}\n")
    dotFile.close()
    logging.info(f"FILE {filename} AGGIORNATO!")
