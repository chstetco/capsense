import pickle
import matplotlib.pyplot as plt
import numpy as np

depth_cap, rgb_cap, rgb_global, cap_vector = pickle.load(open("trial02_assgym.p", "rb"))

plt.ion()

fig, axs = plt.subplots(2, 2)

for k in range(100):
    axs[0, 0].imshow(np.array(rgb_global[k]))
    axs[0, 1].imshow(np.array(rgb_cap[k]))
    axs[1, 0].imshow(np.array(depth_cap[k]))
    axs[1, 1].plot(np.array(cap_vector[0:k]))
    axs[0, 0].title.set_text('Scene View')
    axs[0, 1].title.set_text('RGB Image Cap. Sensor')
    axs[1, 0].title.set_text('Depth Image Cap. Sensor')
    axs[1, 1].title.set_text('Cap. Values (F)')
    plt.pause(0.05)
    plt.show()

