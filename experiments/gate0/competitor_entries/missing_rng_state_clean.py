import json

import torch

torch.manual_seed(20260810)
model = torch.nn.Linear(1, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
losses = []
for _step in range(4):
    optimizer.zero_grad()
    loss = (model(torch.ones(1, 1)) - torch.rand(1, 1)).square().mean()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
print(json.dumps({"losses": losses, "weight": model.weight.detach().flatten().tolist()}))
