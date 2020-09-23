from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import numpy as np


f = open('./back_log.txt', 'r')
x = []
y = []
for l in f.readlines():
    l1 = l.split(',')
    y.append(l1[0])
    x.append([float(x) for x in l1[1:]])
f.close()
x = np.array(x)
y = np.array(y)
embeddings = TSNE(n_jobs=4).fit_transform(x)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()