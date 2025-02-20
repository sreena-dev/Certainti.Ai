
# PyTorch 2.5.1 

## What is PyTorch?
**PyTorch** is an open-source deep learning framework developed by **Meta (Facebook AI)**. It is widely used for:
- **Deep Learning** (Neural Networks, Transformers, Vision, NLP)
- **GPU-Accelerated Computing**
- **Automatic Differentiation (Autograd)**
- **Dynamic Computation Graphs**
- **Research and Production (TorchScript, ONNX)**

---

## Installation
To install the latest stable version of PyTorch:
```sh
pip install torch torchvision torchaudio
```
If `2.5.1` is available, install it using:
```sh
pip install torch==2.5.1
```
To install a GPU version:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
For official installation options, visit [PyTorch's website](https://pytorch.org/get-started/).

---

## Key Features & Modules

### 1. Tensor Operations
PyTorch uses `torch.Tensor` as its core data structure, similar to NumPy arrays but with GPU acceleration.

**Example: Basic Tensor Operations**
```python
import torch

# Creating a tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Basic operations
print("Addition:\n", x + y)
print("Matrix Multiplication:\n", torch.matmul(x, y))
```

---

### 2. Automatic Differentiation (Autograd)
PyTorch enables **automatic differentiation**, which is essential for training neural networks.

**Example: Computing Gradients**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x^3

# Compute gradients
y.backward()
print("Gradient of y w.r.t x:", x.grad)  # dy/dx = 3x^2
```

---

### 3. Building Neural Networks (torch.nn)
PyTorch provides the `torch.nn` module for defining deep learning models.

**Example: Simple Neural Network**
```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 1)  # Fully connected layer

    def forward(self, x):
        return self.fc(x)

# Instantiate the model
model = SimpleNN()
x = torch.tensor([[1.0, 2.0]])
output = model(x)
print("Model Output:", output)
```

---

### 4. Optimizers and Training (torch.optim)
PyTorch provides optimizers to update model parameters.

**Example: Training a Model**
```python
import torch.optim as optim

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy data
x_train = torch.tensor([[1.0, 2.0]], requires_grad=True)
y_train = torch.tensor([[1.0]])

# Training step
optimizer.zero_grad()
output = model(x_train)
loss = criterion(output, y_train)
loss.backward()
optimizer.step()

print("Updated Model Parameters:", list(model.parameters()))
```

---

### 5. GPU Acceleration (CUDA)
PyTorch allows models and tensors to run on **GPUs** for faster computation.

**Example: Moving to GPU**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
print("Tensor on GPU:", tensor)
```

---

### 6. Data Loading (torch.utils.data)
PyTorch provides tools for handling large datasets.

**Example: Creating a DataLoader**
```python
from torch.utils.data import DataLoader, TensorDataset

# Dummy dataset
x_data = torch.randn(100, 2)
y_data = torch.randn(100, 1)
dataset = TensorDataset(x_data, y_data)

# DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterating through batches
for batch in dataloader:
    print(batch)
    break
```

---

### 7. Model Saving & Loading
PyTorch allows saving and loading models easily.

**Example: Saving and Loading a Model**
```python
torch.save(model.state_dict(), "model.pth")  # Save model
model.load_state_dict(torch.load("model.pth"))  # Load model
```

---

## What's New in Latest Versions?
Each new version of PyTorch improves **performance, introduces new features, and optimizes existing functionality**.  
To check for the latest version:
```sh
pip install -U torch
```
Or visit the [PyTorch Release Notes](https://pytorch.org/docs/stable/changelog.html).

---

## Final Thoughts
- **PyTorch is one of the most popular deep learning frameworks.**  
- **It is widely used in research, production, and AI development.**  
- **It provides flexibility, GPU acceleration, and easy debugging.**  
