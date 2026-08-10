import json

import torch

torch.manual_seed(20260810)
model = torch.nn.Linear(1, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
losses = []
for step in range(5):
    optimizer.zero_grad()
    loss = (model(torch.tensor([[float(step + 1)]])) - 1.0).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(float(loss.detach()))
print(json.dumps({"losses": losses, "weight": model.weight.detach().flatten().tolist()}))
