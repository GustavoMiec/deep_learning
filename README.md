```markdown
# Previsão de Corrida com PyTorch 🏃‍♂️💨

Este projeto implementa um modelo simples de **regressão com rede neural** utilizando a biblioteca PyTorch. O objetivo é prever o **tempo de conclusão de uma corrida** com base nas **horas de treino**.

---

## 📚 Bibliotecas Utilizadas

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

---

## 📊 Dados

Os dados representam uma relação entre:

- `X`: **Horas de treino**
- `y`: **Tempo de corrida em minutos**

```python
X = torch.tensor([...])  # Horas de treino
y = torch.tensor([...])  # Tempo de corrida (minutos)
```

---

## 🧠 Arquitetura do Modelo

O modelo é uma rede neural simples com:

- 1 camada linear de entrada com 1 neurônio → 5 neurônios (camada oculta)
- Ativação ReLU
- 1 camada linear de saída com 1 neurônio

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## ⚙️ Treinamento

Configurações:

- **Função de perda**: `MSELoss` (Erro quadrático médio)
- **Otimizador**: `SGD` (Gradiente descendente estocástico) com `lr=0.01`
- **Épocas**: 1000

```python
for epch in range(1000):
    ...
    if epch % 100 == 99:
        print(f'Epoch {epch+1}, Loss: {loss.item()}')
```

---

## 🔮 Previsão

Após o treinamento, o modelo é usado para prever o tempo de corrida baseado em 10 horas de treino:

```python
predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')
```

---

## 📈 Visualização

O gráfico compara os dados reais e as previsões feitas pelo modelo:

- 🔵 **Pontos azuis**: dados reais
- 🔴 **Linha vermelha**: linha de previsão do modelo

```python
plt.scatter(X.numpy(), y.numpy(), label='Dados reais', color='blue')
plt.plot(X.numpy(), y_pred.numpy(), label='Previsão do modelo', color='red')
plt.xlabel('Horas de treino')
plt.ylabel('Tempo de corrida (minutos)')
plt.title('Previsão de corrida com PyTorch')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/32a144bd-3883-4c46-b3ef-1d4fc4760b71)


---

## 📝 Conclusão

Este projeto demonstra como construir e treinar uma rede neural simples em PyTorch para **prever o tempo de corrida** com base em **horas de treino**. Ideal para estudos iniciais de aprendizado de máquina com foco em regressão.
```
