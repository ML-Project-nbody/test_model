import torch
import torch.nn as nn
import torch.optim as optim


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Preprocess the dataset
# Assuming your dataset has columns: 'pos1_x', 'pos1_y', 'pos1_z', 'vel1_x', 'vel1_y', 'vel1_z', 'mass1', 'pos2_x', ..., 'mass3', 'time_step'
scaler = StandardScaler()
data = scaler.fit_transform(df)

# Convert to PyTorch tensors
inputs = torch.tensor(
    data[:, :-9], dtype=torch.float32
)  # All columns except the last 9 for positions at next time step
targets = torch.tensor(
    data[:, -9:], dtype=torch.float32
)  # Last 9 columns for positions at next time step

# Add sequence length dimension (batch_size, sequence_length, input_size)
inputs = inputs.unsqueeze(1)


# Define the RNN model
class PlanetRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PlanetRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters
input_size = (
    10  # 3 positions + 3 velocities + 1 mass for each of the 3 planets + 1 time step
)
hidden_size = 64
output_size = 9  # 3 positions for each of the 3 planets
num_layers = 2
num_epochs = 100
learning_rate = 0.001

# Instantiate the model, loss function, and optimizer
model = PlanetRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# Dummy dataset
# You need to replace this with your actual dataset
# The input tensor shape should be (batch_size, sequence_length, input_size)
# The target tensor shape should be (batch_size, output_size)
inputs = torch.randn(100, 1, input_size)  # 100 examples
targets = torch.randn(100, output_size)  # 100 examples

# Training loop
for epoch in range(num_epochs):
    model.train()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "planet_rnn_model.pth")
