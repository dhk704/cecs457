import numpy  as np
import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 2, size=x.shape)


# plt.scatter(x, y, label='Data Points', color='blue', alpha=0.6)


# mean = np.mean(y)
# plt.axhline(mean, color='red', linestyle='--', label='Mean Line')
# plt.plot (x, np.sin(2*np.pi*x/5), color='green', linestyle='-', label='True Function')
plt.plot(x,y, label='Data Points', color='blue', alpha=0.6)
# variance = np.var(y)
plt.show()
