import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy import array

with open("interpret2.pkl", 'rb') as f:
    data=pickle.load(f)

L=199
def importance():
    data=data.mean(axis=0).mean(axis=0)
    plt.scatter(range(len(data)), data)
    plt.xlabel("Feature #")
    plt.ylabel("Importance")
    plt.savefig("feature_importance.pdf")

def get_freq(sample):
    sample=sample.T
    results=[]
    for row in sample:
        row_f=np.fft.fft(row)[1:L]
        results.append(np.abs(row_f))
    return np.array(results)

results=np.array([get_freq(data[i]) for i in range(data.shape[0])])

results=results.mean(axis=0)
# Plot the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(results, aspect='auto', cmap='Blues', interpolation='nearest', extent=[1,L-1,28,0])
plt.colorbar(label='Feature Importance')
plt.title('Input Feature Importance Heatmap')
plt.xlabel('Frequency')
plt.ylabel('Feature #')
plt.savefig(f"freq_importance_2.pdf")
plt.clf()

freq=results.mean(axis=0)
plt.scatter(range(1,1+len(freq)), freq)
plt.savefig("freq.pdf")