import numpy as np
import os
import struct
import logging
import gzip

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)


def load_mnist(path="/"):
    train_labels_path = os.path.join(path, "train-labels-idx1-ubyte")
    train_images_path = os.path.join(path, "train-images-idx3-ubyte")

    test_labels_path = os.path.join(path, "t10k-labels-idx1-ubyte")
    test_images_path = os.path.join(path, "t10k-images-idx3-ubyte")

    labels_path = [train_labels_path, test_labels_path]
    images_path = [train_images_path, test_images_path]

    labels = []
    images = []

    for path in zip(labels_path, images_path):
        with open(path[0], 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            lb = np.fromfile(lbpath, dtype=np.uint8)
            labels.append(lb)

        with open(path[1], 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images.append(np.fromfile(imgpath, dtype=np.uint8).reshape(len(lb), 784))

    logging.info("CARICAMENTO DATASET MNIST COMPLETATO!")
    return images[0], images[1], labels[0], labels[1]


def load_f_mnist(path="/", kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

