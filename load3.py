import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import SparsePCA, PCA


# Target: generate dataset with shape (N_SAMPLES, SEQ_LEN, N_FEATURE)

EKG_SPLIT_INDEX=299
N_FEATURE=20

SEQ_LENGTH=400
DIRECTORY='../data/plan1'
# Get all PNG files in the directory
FILES = [os.path.join(DIRECTORY, file) for file in os.listdir(DIRECTORY) if file.endswith('.png')]

weeks=[12, 16, 36]
data={}
data[12]=[]
data[16]=[]
data[36]=[]
# 0 health; 1 db
labels={}
for wk in weeks:
    labels[wk]=[]

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

    if "36wk" in filename:
        wk=36
    elif "12wk" in filename:
        wk=12
    else:
        wk=16

    raw=np.transpose(image_array[:EKG_SPLIT_INDEX])
    #transformer = PCA(n_components=N_FEATURE, random_state=0)
    #features=transformer.fit_transform(raw)

    return raw, id, hyp, health, wk


#   ******* ROW INDEX >= 299 IS EKG


#  Main
k=0
for file in FILES:
    k+=1
    print(f"Loading PNG {k}")
    sample,_,hyp, health, wk=load_png(file)
    if hyp==False:
        continue
    if health==True:
        l=0
    else:
        l=1
    for i in range(sample.shape[0]//SEQ_LENGTH):
        data[wk].append(sample[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH])
        labels[wk].append(l)

print("Start PCA")
pca = PCA(n_components=N_FEATURE, random_state=0)
to_fit=[]
for wk in weeks:
    to_fit+= list(data[wk])
to_fit=[seq for sample in to_fit for seq in sample]
pca.fit(to_fit)
for wk in weeks:
    for i, sample in enumerate(data[wk]):
        data[wk][i]=pca.transform(sample)
    data[wk]=np.array(data[wk])

with open("plan3.pkl", 'wb') as f:
    pickle.dump(pca, f)
    pickle.dump(data, f)
    pickle.dump(labels, f)
breakpoint()