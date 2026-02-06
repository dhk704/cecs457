# Student: Daehee Kim
# ID: 033241115

import torch
import matplotlib.pyplot as plt

# Creating line
x = torch.linspace(0, 10, 100)
y = 2*x + 1 + torch.randn(100)*2

plt.plot(x,y,label='Data Points', color='blue', alpha=0.6)

X = torch.stack([x, torch.ones(100,)], dim=1)
a,b = torch.linalg.lstsq(X,y.view(100,1)).solution
print(a.item(), b.item())

plt.plot(x, a*x+b, label='Fitted line', color='red')
plt.show()