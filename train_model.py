import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
# Helper func
def randomly_choose_indices(array):
    """
    Randomly choose 80% of indices from the given array.

    Parameters:
        - array: numpy array or list
            The input array from which to choose the indices.

    Returns:
        - chosen_indices: numpy array
            The randomly chosen indices from the input array.
    """
    # Convert input array to numpy array if not already
    array = np.array(array)

    # Get the total number of indices in the input array
    num_indices = len(array)

    # Calculate the number of indices to choose (80% of total)
    num_chosen = int(0.8 * num_indices)

    # Randomly permute the indices
    permuted_indices = np.random.permutation(num_indices)

    # Select the first num_chosen indices from the permuted indices
    train_indices, test_indices = permuted_indices[:num_chosen], permuted_indices[num_chosen:]

    return train_indices, test_indices


# Denoising Autoencoder Model
class DAutoencoder(nn.Module):
    def __init__(self, encoding_dim, input_dim):
        super(DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.PReLU(),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Linear(256, encoding_dim)
            
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.PReLU(),
            nn.Linear(256,1024),
            nn.PReLU(),
            nn.Linear(1024,input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main(data_path,meta_path):
    #Load Data
    df = pd.read_csv(data_path)
    df = df.drop('Unnamed: 0',axis=1)
    meta_df = pd.read_csv(meta_path)
    target = meta_df.ChemoResponse
    encoded_target = target.map({'Resistant':0, 'Sensitive':1})

    #train test split
    train_indices, test_indices = randomly_choose_indices(np.arange(len(df)))
    train = df.iloc[train_indices,:]

    #fit the scaler
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train)
    test_std= scaler.transform(df.iloc[test_indices,:])

    #train the model
    encoding_dim = 128  # choose an appropriate encoding dimension
    Dautoencoder = DAutoencoder(encoding_dim, input_dim=train.shape[1])

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Dautoencoder.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10000 # specify the number of epochs for training
    test_data = torch.tensor(test_std, dtype=torch.float32).unsqueeze(0)
    input_data = torch.tensor(train_std, dtype=torch.float32).unsqueeze(0)
    for epoch in range(num_epochs):
        # Forward pass
        tmp_data = input_data + torch.randn(input_data.size())
        outputs = Dautoencoder(tmp_data)
        loss = criterion(outputs, input_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            test_output = Dautoencoder(test_data)
            test_loss = criterion(test_output, test_data)
            print(f'test loss: {test_loss}')

        if(epoch) % 1000 == 0:
            torch.save(Dautoencoder.state_dict(), f'model{epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser for Data Path and Meta Path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to meta data file')
    args = parser.parse_args()
    data_path = args.data_path
    meta_path = args.meta_path
    main(data_path, meta_path)






