import json
import math
from pathlib import Path
import random
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch
import pytorch_lightning as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
import os
from ray import air, tune
import typer
from ray.tune.result_grid import ResultGrid
torch.manual_seed(666)


ROOT_DIR = Path(__file__).parent.resolve()
CPU_NUMBER = os.cpu_count()
LABEL_NAME = "error_fc"
class DataModule(LightningDataModule):

    def __init__(self,df:pd.DataFrame, label:str,batch_size:int):

        super().__init__()
        self.df = df
        self.label = label
        
        self.batch_size=batch_size
        self.cols_to_drop = [self.label,'obs','prev','obs_fc','date_lancement','pi','date_cible']
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        
        df_2016_2019 = self.df[self.df.date_cible.dt.year < 2020]
        
        self.df_train, self.df_val = train_test_split(df_2016_2019, test_size=0.33, random_state=42)
        self.df_test = self.df[self.df.date_cible.dt.year == 2020]

        self.std_scaler = StandardScaler()

        # INPUT: prev, weather, calendar variables (centered !)
        # OUTPUT: Not centered FCs ([0,1])
        self.x_train = torch.from_numpy(self.std_scaler.fit_transform(self.df_train.drop(columns=self.cols_to_drop))).float()
        self.x_test = torch.from_numpy(self.std_scaler.transform(self.df_test.drop(columns=self.cols_to_drop))).float()
        self.x_val = torch.from_numpy(self.std_scaler.transform(self.df_val.drop(columns=self.cols_to_drop))).float()
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
        return DataLoader(train_split, shuffle=True, batch_size=self.batch_size)
    def val_dataloader(self):
        val_split = TensorDataset(self.x_val, self.y_val)
        return DataLoader(val_split)
    
    # # Useless
    # def test_dataloader(self):
    #     test_split = TensorDataset(self.x_test, self.y_test)
    #     return DataLoader(test_split)
    
    # # Not working (TODO fix)
    # def predict_dataloader(self):
    #     x_dataset = TensorDataset(self.x_test)[0]
    #     return DataLoader(x_dataset)




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
        # config["seed"] is set deterministically, but differs between training runs
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = Regressor(config)
        trainer = pl.Trainer(
            max_epochs=config['epochs'],
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            callbacks=[
                TuneReportCheckpointCallback(metrics="val/loss", 
                                             on="validation_end")])

        
        trainer.fit(model, datamodule=DataModule(df, label=LABEL_NAME, batch_size=config['batch_size']))


def write_best_model_metrics(obs_type, results:ResultGrid):
    best_result = results.get_best_result()
    metrics = {
        "val":{   
            "loss":best_result.metrics['val/loss']
        }
    }
    metrics_dir = Path(f'./metrics/{obs_type}')
    metrics_dir.mkdir(exist_ok=True, parents=True)
    json.dump(metrics, (metrics_dir/ "metrics.json").open('w'))

def main(obs_type: str = None, num_samples:int = 10,max_concurrent_trials:int=10, cpus_per_trial:int=1, gpus_per_trial:int=0):
    
    
    # Set seed for the search algorithms/schedulers
    random.seed(666)
    np.random.seed(666)
    
    df = pd.read_hdf(f'./features/{obs_type}.hdf')
    dm = DataModule(df, LABEL_NAME,32)
    dm.prepare_data()
    input_dim = dm.x_train.size()[1]
    print(input_dim)
    print(df)
  
    param_space = {
        "input_dim": input_dim,
        "seed": tune.randint(0, 10000),
        "layer_1": tune.choice([2, 4, 8,16]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout_rate": tune.uniform(0.0,0.4),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": tune.choice(range(1,10)),
    }
    scheduler = ASHAScheduler(
        
        grace_period=1,
        #max_t=
    )
    trainable_w_params = tune.with_parameters(train_with_config,  df=df)
    trainable = tune.with_resources(trainable_w_params,resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial})
    
    tuner = tune.Tuner(
        trainable,
        
        tune_config=tune.TuneConfig(
            metric="val/loss",
            mode="min",
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            scheduler=scheduler,
            #time_budget_s=5*60 #20 min
        ),
        run_config=air.RunConfig(
            local_dir=ROOT_DIR / 'ray' ,
            name=obs_type
        ),
        param_space=param_space)
    results = tuner.fit()
    write_best_model_metrics(obs_type, results)
    return 
    

    

if __name__ == "__main__":
    typer.run(main)
