import numpy as np
import pandas as pd
import kagglehub
import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from torchmetrics.functional import accuracy, precision, recall, auroc
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class Vectorized_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype = torch.float32), torch.tensor(self.y[idx], dtype = torch.long)

class Model(pl.LightningModule):
    
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
        self.validation_step_outputs = {
            'val_loss' : [],
            'val_acc' : [],
            'val_prec' : [],
            'val_rec' : []
        }

    def forward(self, x):
        return self.model(x).view(-1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())
        preds = torch.round(y_hat)
        acc = accuracy(preds, y, task="binary")
        prec = precision(preds, y, num_classes=2, task="binary")
        rec = recall(preds, y, num_classes=2, task="binary")
        self.validation_step_outputs['val_loss'].append(loss)
        self.validation_step_outputs['val_acc'].append(acc)
        self.validation_step_outputs['val_prec'].append(prec)
        self.validation_step_outputs['val_rec'].append(rec)
        return {'val_loss': loss, 'val_acc': acc, 'val_prec': prec, 'val_rec': rec}

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.validation_step_outputs['val_loss']).mean()
        avg_acc = torch.tensor(self.validation_step_outputs['val_acc']).mean()
        avg_prec = torch.tensor(self.validation_step_outputs['val_prec']).mean()
        avg_rec = torch.tensor(self.validation_step_outputs['val_rec']).mean()

        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc, prog_bar=True)
        self.log('val_prec', avg_prec)
        self.log('val_rec', avg_rec)
        self.validation_step_outputs = {
            'val_loss': [],
            'val_acc': [],
            'val_prec': [],
            'val_rec': []
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 0.001)

path = kagglehub.dataset_download("kazanova/sentiment140")
print("Path to dataset files:", path)
csv = os.listdir(path)[0]

dataset = pd.read_csv(path+f'/{csv}', names = ['target','ids','date','flag','user','text'])
print(*dataset)
target = dataset['target']
text = dataset['text']

vectorizer = HashingVectorizer(n_features = 2**16)
X_train, X_test, y_train, y_test = train_test_split(text, target, random_state=0, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)

X_train = X_train[:50000]
y_train = y_train[:50000]

X_test = X_test[:5000]
y_test = y_test[:5000]

X_val = X_val[:5000]
y_val = y_val[:5000]

train_vectorized = vectorizer.fit_transform(X_train).toarray()
test_vectorized = vectorizer.transform(X_test).toarray()
val_vectorized = vectorizer.transform(X_val).toarray()


train_dataset = Vectorized_dataset(X_train, y_train)
test_dataset = Vectorized_dataset(X_test, y_test)
val_dataset = Vectorized_dataset(X_val, y_val)

train_data = DataLoader(train_dataset, shuffle = True, batch_size = 32)
test_data = DataLoader(test_dataset, batch_size = 32, shuffle = False)
val_data = DataLoader(val_dataset, batch_size = 32, shuffle = False)

input_dim = train_vectorized.shape[1]
model = Model(input_dim)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    save_top_k=2,
    monitor="val_acc",
    mode="max"
)
early_stopping = EarlyStopping(
    monitor="val_acc",
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode="max"
)

trainer = pl.Trainer(
    max_epochs=1000,
    callbacks=[early_stopping, checkpoint_callback],
    log_every_n_steps=1,
    deterministic=True
)

trainer.fit(model, train_data, val_data)