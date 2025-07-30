import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("weights.pkl", "rb") as f:
    all_weights = pickle.load(f)

# Plot the first element of the weight matrix over epochs
weights_to_plot = [w[0, 0] for w in all_weights]

plt.plot(weights_to_plot)
plt.xlabel("Epoch")
plt.ylabel("Weight value")
plt.title("Sampled weight element over epochs")
plt.savefig("weights.png")
print("Plot saved to weights.png")