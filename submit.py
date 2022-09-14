import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from train import Regressor, DataModule, LABEL_NAME
from features import ALLOWED_HORIZONS
import torch
from ray import tune

OBS_TYPES = ['photovoltaique','eolien','consommation','consommation_residuelle']
SUBMISSION_COLS = ['date_cible','date_lancement','quantile_niveau','type','prev_q']


def prepare_submission(obs_type, results, df_type='test'):
    # TODO fix error here
    net=Regressor(results.best_config)

    # Predict quantiles - using dm.predict_loader would be cleaner but does not work....
    with results.best_checkpoint.as_directory() as loaded_checkpoint_dir:
        ckp = torch.load(Path(loaded_checkpoint_dir) / "checkpoint")
        net.load_state_dict(ckp['state_dict'])
    
    # Load data
    OG_df = pd.read_hdf(f'./features/{obs_type}.hdf')
    dm = DataModule(OG_df, label=LABEL_NAME, batch_size=results.best_config['batch_size'])
    dm.prepare_data()
    x = getattr(dm, "x_"+df_type)
    df = getattr(dm, "df_"+df_type)
    
    # Predict
    net.eval()
    outs = net(x).detach()
    
    quantiles_cols = [f"{level:.3f}" for level in  np.array(net.quantile_levels)]
    quantiles_df = pd.DataFrame(columns=quantiles_cols, data=outs)
    
    # Concat to original DF
    results_df = pd.concat([df, quantiles_df.set_index(df.index)],axis=1)
    
    # Remove useless echeances
    results_df = results_df[results_df.echeance.isin(ALLOWED_HORIZONS)]
    
    # Large to long
    results_df['id'] = results_df.index
    COLS_TO_KEEP = ['date_cible','date_lancement','pi','echeance','prev','obs']
    for col in quantiles_cols:
        results_df[col] += results_df['prev_fc']
    results_df = results_df[COLS_TO_KEEP+quantiles_cols]
    results_df =  pd.melt(results_df, id_vars=COLS_TO_KEEP,value_vars=quantiles_cols,var_name="quantile_niveau",value_name="prev_q")
    results_df['quantile_niveau'] = pd.to_numeric(results_df['quantile_niveau'])
    results_df['type'] = obs_type
    
    # Multiply by installed power 
    results_df['prev_q'] = results_df['prev_q'] * results_df['pi']
    results_df.drop(columns='pi', inplace=True)
    # Zeroing negative productions
    results_df.loc[results_df.prev_q < 0, 'prev_q'] = 0 
    return results_df

def main():
    
    outs = []
    for obs_type in OBS_TYPES:
        
        # Only run 
        exp_path_list= list(Path(f"./ray/{obs_type}/").glob('*experiment*'))
        assert len(exp_path_list) == 1
        exp_path = exp_path_list[0]
        
        results = tune.ExperimentAnalysis(experiment_checkpoint_path=exp_path,default_metric="val/loss",default_mode="min")
        print(f"""
            Preparing submission for {obs_type}...
            Using Experiment {exp_path}
            Validation loss: {results.best_result['val/loss']}
            """)
        

        outs.append(prepare_submission(obs_type, results))
        submission = pd.concat(outs,axis=0)
        submission.reset_index(inplace=True)
        submissions_dir = Path('./submissions/')
        submissions_dir.mkdir(parents=True, exist_ok=True)
        submission[SUBMISSION_COLS].to_feather(submissions_dir / "AR.feather", compression="zstd")
if __name__ == "__main__":
    main()