import pickle

import torch
import seaborn as sns
import matplotlib.pyplot as plt

from utils import heart_rate, model2, model3, predict_1


if torch.cuda.is_available()==False:
    model2.load_state_dict(torch.load(f'model2.pth', map_location=torch.device('cpu')))
    model3.load_state_dict(torch.load(f'model3.pth', map_location=torch.device('cpu')))
else:
    model2.load_state_dict(torch.load(f'model2.pth'))
    model3.load_state_dict(torch.load(f'model3.pth'))
model2.eval()  # Set the model to evaluation mode
model3.eval()  # Set the model to evaluation mode

# Load PCA transformer
with open("plan2.pkl", 'rb') as f:
    pickle.load(f)
    pickle.load(f)
    pca2=pickle.load(f)

with open("plan3.pkl", 'rb') as f:
    pca3=pickle.load(f)

# Load data and transform
# data is inhomogeneous
with open("plan_rate.pkl", 'rb') as f:
    data=pickle.load(f)
    rates=pickle.load(f)

def main(model, pca, output):
    s=[[],[]]
    for i, sample in enumerate(data):
        data[i]=pca.transform(sample)
    print("Transformed")
    for sample, rate in zip(data, rates):
        s[predict_1(sample, model)].append(rate)
        breakpoint()


    plt.hist(s[0],density=True, alpha=0.5)
    sns.kdeplot(data, color='blue', label="Predicted Healthy")

    plt.hist(s[1],density=True, alpha=0.5)
    sns.kdeplot(data, color='red', label="Predicted Diseased")

    plt.legend()
    plt.title(f"Model {output}")
    plt.savefig(f"rate{output}.pdf")
    plt.clf()

main(model2, pca2, 2)
main(model3, pca3, 3)
    