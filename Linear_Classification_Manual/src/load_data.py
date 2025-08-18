import struct
import numpy as np

with open("src/data/train-images-idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    print("Train Images:", num_images, "Boyut:", rows, "x", cols)
    train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)


with open("src/data/train-labels-idx1-ubyte", "rb") as f:
    magic, num_labels = struct.unpack(">II", f.read(8))
    print("Train Labels:", num_labels)
    train_labels = np.frombuffer(f.read(), dtype=np.uint8)

with open("src/data/t10k-images-idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    print("Test Images:", num_images, "Boyut:", rows, "x", cols)
    test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)


with open("src/data/t10k-labels-idx1-ubyte", "rb") as f:
    magic, num_labels = struct.unpack(">II", f.read(8))
    print("Test Labels:", num_labels)
    test_labels = np.frombuffer(f.read(), dtype=np.uint8)

np.save("src/data/train_images.npy", train_images)
np.save("src/data/train_labels.npy", train_labels)
np.save("src/data/test_images.npy", test_images)
np.save("src/data/test_labels.npy", test_labels)