import pandas as pd
from pathlib import Path
import datetime
from vacances_scolaires_france import SchoolHolidayDates

calendar = SchoolHolidayDates()

ALLOWED_HORIZONS = [0.5,1,2,4]

other_cols = ["tcc","t2m","ssrd","ff100","u100","v100"]
data_folder = Path("data_challenge/data")
feat_dir = Path("features")

def percent_rows_na(df):
    return (len(df)-len(df.dropna(axis=0)))*100/len(df)


def assign_meteo_date_lancement(dt):    
    
    if 0<= dt.hour < 6:
         hour_date_lancement = 0
    elif 6< dt.hour <= 12:
         hour_date_lancement = 6
    elif 12< dt.hour <= 18:
         hour_date_lancement = 12
    else:
         hour_date_lancement = 18
    return dt.replace(hour=hour_date_lancement)
         
         
def add_holiday_and_weekends_indicators(df):
    # WARNING : use only for conso !
    df['is_weekend'] = (df.date_cible.dt.dayofweek > 4).astype(int).astype(float)
    # df['date'] = df.date_cible.dt.date
    # my_holiday_func = lambda zone, date : calendar.is_holiday_for_zone(date, zone)
    # df['is_holiday_A'] = df.date.apply(partial(my_holiday_func,'A')).astype(int).astype(float)
    # df['is_holiday_B'] = df.date.apply(partial(my_holiday_func,'B')).astype(int).astype(float)
    # df['is_holiday_C'] = df.date.apply(partial(my_holiday_func,'C')).astype(int).astype(float)
    # df.drop(columns='date',inplace=True)
    return df
    
def fix_echeance(df):
    df['echeance'] = (df.date_cible - df.date_lancement).dt.seconds/3600

def remove_useless_horizons(df):
    return df.loc[df.echeance.isin(ALLOWED_HORIZONS)]

def add_datetime_features(df):
    # time in the year
    #df['year_dt'] =  datetime.datetime(year=df.date_cible.dt.year)
    tzinfo = df.date_cible.dt.tz
    df['tiy'] = (df.date_cible - df.date_cible.dt.year.apply(lambda y: datetime.datetime(year=y,month=1,day=1,tzinfo=tzinfo))).dt.total_seconds()/(365*24*60*60)
    # time in the day
    df['tid'] = (df.date_cible.dt.hour *3600 + df.date_cible.dt.minute *60 + df.date_cible.dt.second)/(24*60*60)
    # TODO: type of day for consumption

def main():
    print('running')
    #df_prodpv_fc_q90 = pd.read_feather(os.path.join(data_folder, "productionPV_FC_cielclair_q90.feather"))
    #df_list_station = pd.read_csv(os.path.join(data_folder,"liste_stations.csv"), sep=";", header=0)
    feat_dir.mkdir(exist_ok=True)
    df_prev_sans_obs2020 = pd.read_feather( data_folder/ "df_prev_sans_obs2020.feather")
    print(df_prev_sans_obs2020.echeance.unique()) # echeance 30min - 7h
    print(df_prev_sans_obs2020.isnull().sum()) # Missing 417844 observations (for 2020)
    # Add fake PI for conso in order to get a FC between 0 and 1
    df_prev_sans_obs2020.loc[df_prev_sans_obs2020.type.str.contains('conso'),'pi'] = df_prev_sans_obs2020[df_prev_sans_obs2020.type.str.contains('conso')].obs.max() + 10**5
    # Compute FCs
    df_prev_sans_obs2020['obs_fc'] = df_prev_sans_obs2020['obs'] / df_prev_sans_obs2020['pi']
    df_prev_sans_obs2020['prev_fc'] = df_prev_sans_obs2020['prev'] / df_prev_sans_obs2020['pi']
    df_prev_sans_obs2020['error_fc'] = df_prev_sans_obs2020['obs_fc'] - df_prev_sans_obs2020['prev_fc']

    add_datetime_features(df_prev_sans_obs2020)
    
    
    df_grille_zoneclimat_fin18 = pd.read_feather(data_folder /"grille_zone_climatique_fin2018.feather")
    df_grille_zoneclimat_fin18.head(10)
    df_meteo_zone_eol = pd.read_feather(data_folder / "meteo_zone_echeance12_2016_2020_HRES_piEOL_smooth.feather")
    df_meteo_zone_eol.groupby('date_lancement_meteo').count()

    df_meteo_zone_eol = pd.read_feather(data_folder/ "meteo_zone_echeance12_2016_2020_HRES_piEOL_smooth.feather")
    print(sorted(df_meteo_zone_eol.echeance.unique())) # echeance 0min - 11h30
    assert df_meteo_zone_eol.isnull().sum().sum() == 0 # No missing value
    
    # Long to large
    other_cols = ["tcc","t2m","ssrd","ff100","u100","v100"]
    df_meteo_zone_eol = df_meteo_zone_eol.pivot(index=["date_lancement_meteo","date_cible","echeance"], values=other_cols, columns="zone").reset_index()
    assert df_meteo_zone_eol.isnull().sum().sum() == 0 # No missing value
    

    df_meteo_zone_pv = pd.read_feather(data_folder / "meteo_zone_echeance12_2016_2020_HRES_piPV_smooth.feather")
    print(f"echeances:{sorted(df_meteo_zone_pv.echeance.unique())}") # echeance 0min - 11h30
    print(f"zones:{sorted(df_meteo_zone_pv.zone.unique())}") # echeance 0min - 11h30
    assert df_meteo_zone_pv.isnull().sum().sum() == 0 # No missing value


    # Long to large
    df_meteo_zone_pv = df_meteo_zone_pv.pivot(index=["date_lancement_meteo","date_cible","echeance"], values=other_cols, columns="zone").reset_index()
    assert df_meteo_zone_pv.isnull().sum().sum() == 0 # No missing value
    

    df = df_prev_sans_obs2020

    # Drop uselss horizons
    df = df[df.echeance.isin([0.5,1,2,4])]

    df['date_lancement_meteo'] = df.date_lancement.apply(assign_meteo_date_lancement)
    df_pv = df[df.type =='photovoltaique'].drop(columns='type')
    df_conso = df[df.type =='consommation'].drop(columns='type')
    df_conso_res = df[df.type =='consommation_residuelle'].drop(columns='type')
    df_eol = df[df.type =='eolien'].drop(columns='type')
    # No missing data in year < 2020, prev
    assert percent_rows_na(df_eol[df_eol.date_cible.dt.year<2020])==0.0 # No missing value in train
    assert percent_rows_na(df_pv[df_pv.date_cible.dt.year<2020])==0.0 # No missing value in train
    assert percent_rows_na(df_conso_res[df_conso_res.date_cible.dt.year<2020])==0.0 # No missing value in train
    assert percent_rows_na(df_conso[df_conso.date_cible.dt.year<2020])==0.0 # No missing value in train

    # Add holidays for conso
    df_conso = add_holiday_and_weekends_indicators(df_conso) 
    df_conso_res = add_holiday_and_weekends_indicators(df_conso_res) 
    print(df_conso)


    # Merge with weather data
    
    # PV
    PV_USELESS_COLS = ['ff100','u100','v100']
    df_pv_meteo = df_pv.merge(df_meteo_zone_pv.drop(columns=PV_USELESS_COLS+['echeance']), on=['date_cible','date_lancement_meteo'], how='inner').drop(columns='date_lancement_meteo')

    
    # LONG
    EOL_USELESS_COLS = ['tcc','ssrd','t2m']
    df_eol_meteo = df_eol.merge(df_meteo_zone_eol.drop(columns=EOL_USELESS_COLS+['echeance']), on=['date_cible','date_lancement_meteo'], how='inner').drop(columns='date_lancement_meteo')
    # TODO check how many values are lost during inner join
    

    # WARNING: TODO USE REAL WEATHER DATA 
    df_conso_meteo = df_conso.merge(df_meteo_zone_pv.drop(columns=PV_USELESS_COLS+['echeance']), on=['date_cible','date_lancement_meteo'], how='inner').drop(columns='date_lancement_meteo')
    
    df_conso_res_meteo = df_conso_res.merge(df_meteo_zone_pv.drop(columns=PV_USELESS_COLS+['echeance']), on=['date_cible','date_lancement_meteo'], how='inner').drop(columns='date_lancement_meteo')
    
    # Save data
    df_pv_meteo.to_hdf(feat_dir / "photovoltaique.hdf",key="data")
    df_eol_meteo.to_hdf(feat_dir / "eolien.hdf",key="data")
    df_conso_meteo.to_hdf(feat_dir / "consommation.hdf",key="data")
    df_conso_res_meteo.to_hdf(feat_dir / "consommation_residuelle.hdf",key="data")
    
if __name__ == '__main__':
    main()