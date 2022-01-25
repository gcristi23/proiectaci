import json
import matplotlib.pyplot as plt

models = ['deep_voting', 'voting', 'autoencoder']

for model in models:
    with open(f"{model}_train_loss") as f:
        data = json.load(f)
    plt.figure()
    plt.title(model)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(data)
plt.show()