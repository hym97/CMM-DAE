# README

  
## 1. Set up the env
We first need to create the env by

`conda env create -f environment.yml`

  

Then activate the env by

`conda activate CMM`

  

## 2. Train the model yourself

Please use `train_model.py` to train the model. 
### Note that I did not specify the `device` in that file. 
To train the model
`python train_model --data_path DATA_path(i.e. BR0_data.csv) --meta_path(i.e. BR0_meta.csv)`


  ## 3. Use pretrained model

One may use the pretrained scaler and model (trained over 10,000 epochs):
1.  Get the scaler and model params from [CMM DAE](https://drive.google.com/drive/u/0/folders/1_cJae0CXtsGWe1h7Q26Y-CHkAH7BcncF)
2. Use the following instructions (please modify the path):
`python getData --data_path .\BR0_data.csv --scaler_path .\scaler.joblib --model_path .\DAE.pth --output_path D:\cm\CMM-Proj`
3. We will get `output.npy` as the output, whose dims is (input.shape[0], 128).  