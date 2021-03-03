from cnn_preprocess import CNNPreprocess
from cnn import CNN
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

file_sets = [
    (["./challenge_data/training-a/a" + str(i).zfill(4) for i in range(1, 410)], "./challenge_data/training-a/"),
    (["./challenge_data/training-b/b" + str(i).zfill(4) for i in range(1, 491)], "./challenge_data/training-b/"),
    (["./challenge_data/training-c/c" + str(i).zfill(4) for i in range(1, 32)], "./challenge_data/training-c/"),
    (["./challenge_data/training-d/d" + str(i).zfill(4) for i in range(1, 56)], "./challenge_data/training-d/"),
    (["./challenge_data/training-e/e" + str(i).zfill(5) for i in range(1, 2142)], "./challenge_data/training-e/"),
    (["./challenge_data/training-f/f" + str(i).zfill(4) for i in range(1, 115)], "./challenge_data/training-f/"),
]

cnn_preprocess = CNNPreprocess(file_sets=file_sets)
data = cnn_preprocess.preprocess_data()
X_orig = np.array(data["values"])
y = np.array(data["labels"])

nsample, nx, ny, nz = X_orig.shape
X = X_orig.reshape((nsample, nx*ny*nz))

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, edgecolor='none', alpha=0.5)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()