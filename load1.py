import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import SparsePCA, PCA


# Target: generate dataset with shape (N_SAMPLES, SEQ_LEN, N_FEATURE)

EKG_SPLIT_INDEX=299
N_FEATURE=EKG_SPLIT_INDEX//10

SEQ_LENGTH=400
DIRECTORY='../data/plan1'
# Get all PNG files in the directory
FILES = [os.path.join(DIRECTORY, file) for file in os.listdir(DIRECTORY) if file.endswith('.png')]

data=[]
# 0 health; 1 db
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



def info():
    x=load_png(FILES[0])
    a=x[0]
    for i, row in enumerate(a):
        flag=True
        for k in row:
                if k!=0:
                    flag=False
                    break
        if flag:
            print(i)
        #break

#   ******* ROW INDEX >= 299 IS EKG


#  Main
k=0
for file in FILES:
    k+=1
    print(f"Loading PNG {k}")
    sample,_,hyp, health=load_png(file)
    if hyp==False:
        continue
    if health==True:
        l=0
    else:
        l=1
    for i in range(sample.shape[0]//SEQ_LENGTH):
        data.append(sample[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH])
        labels.append(l)

with open("plan1.pkl", 'wb') as f:
    data=np.array(data)
    pickle.dump(data, f)
    pickle.dump(labels, f)
breakpoint()