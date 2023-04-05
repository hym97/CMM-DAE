import torch
import torch.nn as nn
import pandas as pd
from joblib import load
import argparse
import numpy as np

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



def main(data_path, scaler_path, model_path, output_path):
    # Your code here that uses the arguments
    print(f"Data path: {data_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Model path: {model_path}")
    
    #Load data
    df = pd.read_csv(data_path)
    df = df.drop('Unnamed: 0', axis=1)
    
    #Load scaler
    scaler = load(scaler_path)
    scaled_X = scaler.transform(df)

    #Load Model
    model = DAutoencoder(128, scaled_X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    encoder = model.encoder
    input_data = torch.tensor(scaled_X, dtype=torch.float32).unsqueeze(0)

    output = encoder(input_data).squeeze(0).detach().numpy()
    np.save(f'{output_path}/output.npy', output)



if __name__ == '__main__':

# Define the arguments
    parser = argparse.ArgumentParser(description='Perform Dimensionality Reduction via DAE.')
    parser.add_argument('--data_path', metavar='data_path', type=str, help='path to BR0_data.csv file')
    parser.add_argument('--scaler_path', metavar='scaler_path', type=str, help='path to scaler file')
    parser.add_argument('--model_path', metavar='model_path', type=str, help='path to model file')
    parser.add_argument('--output_path', type=str, help='path to output')

    args = parser.parse_args()
    main(args.data_path, args.scaler_path, args.model_path, args.output_path)