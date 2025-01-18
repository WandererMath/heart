import pickle

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

with open("plan2.pkl", 'rb') as f:
    pickle.load(f)
    pickle.load(f)
    pca=pickle.load(f)

for k in range(28):


    coeff=pca.components_
    target_pc=coeff[k]
    plt.plot(target_pc)
    plt.xlabel("Blood Flow Velocity (From High to Low to Negative)")
    plt.ylabel("PCA Loadings")
    plt.title(f"{k}-th Principal Component")
    plt.savefig(f"pc/pc_{k}.pdf")
    plt.clf()