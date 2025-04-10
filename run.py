import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

model = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()  # <- Aqui foi corrigido

    if epch % 100 == 99:
        print(f'Epoch {epch+1}, Loss: {loss.item()}')

with torch.no_grad():
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')


import matplotlib.pyplot as plt

# Dados originais
plt.scatter(X.numpy(), y.numpy(), label='Dados reais', color='blue')

# Previsões do modelo
with torch.no_grad():
    y_pred = model(X)
plt.plot(X.numpy(), y_pred.numpy(), label='Previsão do modelo', color='red')

plt.xlabel('Horas de estudo')
plt.ylabel('Nota')
plt.title('Regressão com PyTorch')
plt.legend()
plt.grid(True)
plt.show()
