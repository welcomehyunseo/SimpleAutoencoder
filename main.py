import torch

print('Hello, World!')

# 1. Set random seed for reproducibility
torch.manual_seed(42)

d_model = 4

# 2. Create synthetic dataset
# 100 samples, each with 4 features. 
# Because input and predictions are the same, our target is simply X.
X = torch.randn(100, d_model)

# 3. Define the Autoencoder Model
class SimpleAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder: Compresses 4 features down to 2 latent dimensions

        llm_depth = 10
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d_model, llm_depth),
            torch.nn.LeakyReLU()
        )
        # Decoder: Reconstructs the 2 dimensions back to 4 features (Predictions)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(llm_depth, d_model),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model, loss function, and optimizer
model = SimpleAutoencoder()
criterion = torch.nn.MSELoss()                  # Mean Squared Error measures reconstruction error
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop (Backpropagation)
epochs = 5000
print("Starting Training...")

for epoch in range(epochs):
    # Forward Pass: Generate predictions from the input
    predictions = model(X)
    
    # Calculate Loss: Compare predictions directly against the input (X)
    loss = criterion(predictions, X)
    
    # Backpropagation
    optimizer.zero_grad()   # Clear previous gradients
    loss.backward()         # Computes gradients across all weights via chain rule
    optimizer.step()        # Update model weights using gradient descent
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. Testing the Model
model.eval()
with torch.no_grad():

    sample_input = torch.randn(1, d_model)
    predicted_output = model(sample_input)
    
    print("\n--- Results Comparison ---")
    print("Original Input:\n", sample_input)
    print("\nModel Prediction (Reconstruction):\n", predicted_output)

