import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
import torch
import pandas as pd
import numpy as np
import tqdm
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
accelerator = Accelerator()


device = accelerator.device if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
songs=pd.read_csv('/content/song_recommendation_dataset_real_5000.csv')
songs.head(10)
le_time = LabelEncoder()
songs['time_of_day'] = le_time.fit_transform(songs['time_of_day'])  

le_genre = LabelEncoder()
songs['genre/mood'] = le_genre.fit_transform(songs['genre/mood'])  


le_music = LabelEncoder()
songs['music'] = le_music.fit_transform(songs['music'])  

print(songs)
X = songs[['time_of_day', 'steps', 'temperature', 'wind_speed', 'genre/mood']].to_numpy(dtype=np.float32)
y = songs['music'].to_numpy(dtype=np.int64) 

print("X dtype:", X.dtype)  
print("y dtype:", y.dtype)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32) 
y_train = torch.tensor(y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)  
y_test = torch.tensor(y_test, dtype=torch.int64)

print("X_train:", X_train)
print("y_train:", y_train)
print("X_train:\n", X_train)
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork()
print(model)


loss_fn = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
n_epochs = 100  
batch_size = 10  
batch_start = torch.arange(0, len(X_train), batch_size)
 

best_mse = np.inf   
best_weights = None
history = []
 
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
          
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
           
            bar.set_postfix(mse=float(loss))
   
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
 

model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()
 
model.eval()
with torch.no_grad():
   
    for i in range(5):
        X_sample = X_test_raw[i: i+1]
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
        print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
