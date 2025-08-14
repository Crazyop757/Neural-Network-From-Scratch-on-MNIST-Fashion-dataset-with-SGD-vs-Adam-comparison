import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sgd_accs = np.load("sgd_accs.npy")
adam_accs = np.load("adam_accs.npy")

fig, ax = plt.subplots()
ax.set_xlim(0, len(sgd_accs))
ax.set_ylim(0, 1)
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.set_title("Optimizer Horse Race 🐎")
sgd_line, = ax.plot([], [], 'r-', label="SGD 🐎", lw=3)
adam_line, = ax.plot([], [], 'b-', label="Adam 🐎", lw=3)
ax.legend()

def update(frame):
    sgd_line.set_data(range(frame+1), sgd_accs[:frame+1])
    adam_line.set_data(range(frame+1), adam_accs[:frame+1])
    return sgd_line, adam_line

ani = FuncAnimation(fig, update, frames=len(sgd_accs), interval=500, blit=True)
plt.show()
