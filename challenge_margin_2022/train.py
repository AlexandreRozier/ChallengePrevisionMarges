from pathlib import Path
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torch
import pytorch_lightning as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import os
from ray.tune.schedulers import ASHAScheduler

import typer

ROOT_DIR = Path(__file__).parent.resolve()
CPU_NUMBER = os.cpu_count()
class DataModule(LightningDataModule):

    def __init__(self,df:pd.DataFrame, label:str,batch_size:int):

        super().__init__()
        self.df = df
        self.label = label
        
        self.batch_size=batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        
        df_2016_2019 = self.df[self.df.date_cible.dt.year < 2020]
        
        self.df_train, self.df_val = train_test_split(df_2016_2019, test_size=0.33)
        self.df_test = self.df[self.df.date_cible.dt.year == 2020]

        self.std_scaler = StandardScaler()

        self.x_train = torch.from_numpy(self.std_scaler.fit_transform(self.df_train.drop(columns=[self.label,'obs','date_lancement','pi','date_cible']))).float()
        self.x_test = torch.from_numpy(self.std_scaler.transform(self.df_test.drop(columns=[self.label,'obs','date_lancement','pi','date_cible']))).float()
        self.x_val = torch.from_numpy(self.std_scaler.transform(self.df_val.drop(columns=[self.label,'obs','date_lancement','pi','date_cible']))).float()
        self.y_train = torch.from_numpy(self.df_train[self.label].values).float()
        self.y_test = torch.from_numpy(self.df_test[self.label].values).float()
        self.y_val = torch.from_numpy(self.df_val[self.label].values).float()
        # sanity check
        assert not torch.isnan(self.x_train).any()
        assert not torch.isnan(self.x_test).any()
        assert not torch.isnan(self.x_val).any()
        assert not torch.isnan(self.y_train).any()
        assert not torch.isnan(self.y_val).any()

    def train_dataloader(self):
        train_split = TensorDataset(self.x_train, self.y_train)
        return DataLoader(train_split, shuffle=True, batch_size=self.batch_size,num_workers=CPU_NUMBER)
    def val_dataloader(self):
        val_split = TensorDataset(self.x_val, self.y_val)
        return DataLoader(val_split,num_workers=CPU_NUMBER)
    
    # Useless
    def test_dataloader(self):
        test_split = TensorDataset(self.x_test, self.y_test)
        return DataLoader(test_split,num_workers=CPU_NUMBER)
    
    def predict_dataloader(self):
        x_dataset = TensorDataset(self.x_test)[0]
        return DataLoader(x_dataset,num_workers=1)




class Regressor(pl.LightningModule):
    
    def __init__(self, config): 
        super(Regressor, self).__init__()

        self.lr = config["lr"]
        self.dropout_rate = config["dropout_rate"]
        # TODO n_hidden, n_neurons
        
        layer_1_dim = config["layer_1"]
        self.batch_size = config["batch_size"]
        self.quantile_levels = torch.arange(0.005,1.00,0.005,dtype=torch.float32)
        self.quantile_weights = 1/(self.quantile_levels*(1-self.quantile_levels))
        # Input shape is (batch_size,  n_dim)
        self.layer_1 = torch.nn.Linear(config['input_dim'], layer_1_dim)
        self.drop_1 = torch.nn.Dropout(p=self.dropout_rate)
        self.layer_2 = torch.nn.Linear(layer_1_dim, self.quantile_levels.size(0))
  
    def forward(self, x):
        x = self.drop_1(torch.relu(self.layer_1(x)))
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch    
        y_hat = self.forward(x)
        
        loss = self.custom_quantile_loss(y_hat, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.custom_quantile_loss(y_hat, y)

        self.log('val/loss', loss,on_step=False, on_epoch=True)

        return loss
    


    def custom_quantile_loss(self, y_hat, y):
        
        y = torch.unsqueeze(y,1).repeat(1, self.quantile_levels.size(0))
        
        quantile_loss_tensor = (y_hat-y)*(indicator(y_hat-y)-self.quantile_levels)*self.quantile_weights
        return torch.mean(quantile_loss_tensor)

# Code Jean: 
# diff <- y - q_hat
# mean((diff*tau*(diff>0)-diff*(1-tau)*(diff<0))/(tau*(1-tau))) -> OK
def indicator(e):
    return 1 * torch.gt(e, 0) 

def train_with_config(config, df=None):
        model = Regressor(config)
        trainer = pl.Trainer(
            max_epochs=config['epochs'],
            accelerator='cpu',
            enable_progress_bar=False,
            devices=1,
            enable_checkpointing=True,
            callbacks=[
                TuneReportCheckpointCallback(metrics="val/loss", 
                                             on="validation_end")])

        
        trainer.fit(model, datamodule=DataModule(df, label='fc', batch_size=config['batch_size']))
        return trainer


def main(obs_type: str = None, num_samples:int = 10,max_concurrent_trials:int=10, cpus_per_trial:int=1, gpus_per_trial:int=0):
    
    
    
    df = pd.read_hdf(f'./features/{obs_type}.hdf')
    print(df)
  
    config = {
        "input_dim": 38,
        "layer_1": tune.choice([2, 4, 8,16]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout_rate": tune.uniform(0.0,0.4),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": tune.choice(range(1,10)),
    }
    scheduler = ASHAScheduler(
        metric='val/loss',
        mode="min",
        grace_period=1,
    )
    trainable = tune.with_parameters(
        train_with_config,  df=df)
    tune.run(
        trainable,
        scheduler=scheduler,
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        config=config,
        num_samples=num_samples,
        max_concurrent_trials= max_concurrent_trials,
        name=ROOT_DIR / 'ray' / obs_type)

if __name__ == "__main__":
    typer.run(main)