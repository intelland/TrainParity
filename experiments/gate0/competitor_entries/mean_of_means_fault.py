import json

import torch

torch.manual_seed(20260810)
model = torch.nn.Linear(1, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[1.0], [1.0], [1.0], [8.0]])
optimizer.zero_grad()
loss = ((model(x[:1]) - y[:1]).square().mean() + (model(x[1:]) - y[1:]).square().mean()) / 2
loss.backward()
optimizer.step()
print(json.dumps({"loss": float(loss.detach()), "weight": model.weight.detach().flatten().tolist()}))
