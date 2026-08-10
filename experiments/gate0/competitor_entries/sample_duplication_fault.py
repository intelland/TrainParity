import json

import torch

torch.manual_seed(20260810)
model = torch.nn.Linear(1, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
sample_ids = [0, 1, 2, 2]
losses = []
for sample_id in sample_ids:
    x = torch.tensor([[float(sample_id + 1)]])
    optimizer.zero_grad()
    loss = (model(x) - x * 0.5).square().mean()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
print(json.dumps({"losses": losses, "sample_ids": sample_ids, "weight": model.weight.detach().flatten().tolist()}))
