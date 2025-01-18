import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import SparsePCA, PCA

from utils import heart_rate

# Target: generate dataset with shape (N_SAMPLES, SEQ_LEN, N_FEATURE)

EKG_SPLIT_INDEX=299
N_FEATURE=EKG_SPLIT_INDEX//10

SEQ_LENGTH=400
DIRECTORY='../data/plan1'
# Get all PNG files in the directory
FILES = [os.path.join(DIRECTORY, file) for file in os.listdir(DIRECTORY) if file.endswith('.png')]

data=[]
# Heart Rate
labels=[]

def load_png(filename):
    # Load the PNG file
    image = plt.imread(filename)

    # Convert the image to a NumPy array (already handled by plt.imread)
    image_array = np.array(image)

    filename=filename.split("/")[-1]
    if "hyperemia" in filename:
        hyp=True
    else:
        hyp=False
    if "het" in filename:
        health=True
    else:
        health=False
    id=int(filename.split('_')[0])

    raw=np.transpose(image_array[:EKG_SPLIT_INDEX])
    #transformer = PCA(n_components=N_FEATURE, random_state=0)
    #features=transformer.fit_transform(raw)

    return raw, id, hyp, health


#  Main
k=0
for file in FILES:
    k+=1
    print(f"Loading PNG {k}")
    sample,_,hyp, health=load_png(file)
    if hyp==False:
        continue
    data.append(sample)
    t=heart_rate(file)
    labels.append(t)
    print(t)


with open("plan_rate.pkl", 'wb') as f:

    pickle.dump(data, f)
    pickle.dump(labels, f)
#breakpoint()