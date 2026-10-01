import torch
from loss import NaVIPLoss


# Create the loss function
loss_function = NaVIPLoss()

# Dummy predicted values
predicted_position = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0]
])

predicted_orientation = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.1, 0.1]
])

# Dummy ground truth
true_position = torch.tensor([
    [1.2, 2.1, 3.1],
    [2.1, 2.9, 4.2]
])

true_orientation = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.1, 0.1]
])

# Calculate loss
loss = loss_function(
    predicted_position,
    predicted_orientation,
    true_position,
    true_orientation
)

print("Loss:", loss.item())
print("sx:", loss_function.sx.item())
print("sq:", loss_function.sq.item())
print("Loss is valid:", torch.isfinite(loss).item())