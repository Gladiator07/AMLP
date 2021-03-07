import numpy as np
import matplotlib.pyplot as plt

random_image = np.random.randint(0, 256, (25, 256))

plt.figure(figsize=(7,7))
plt.imshow(random_image, cmap='gray', vmin=0, vmax=255)
plt.savefig("../plots/random_image.png")