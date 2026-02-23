from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

with open("model_price.pkl",  "rb") as f: price_model  = pickle.load(f)
with open("model_demand.pkl", "rb") as f: demand_model = pickle.load(f)
with open("model_spike.pkl",  "rb") as f: spike_model  = pickle.load(f)

WIND_Q25 = 544.9575
NET_Q90  = 5065.9212
PRICE_SHIFT = 462.5

class RawInput(BaseModel):
    # Time
    hour_of_day: int
    day_of_week: int
    month: int
    # Market
    price: float
    price_max: float
    price_min: float
    demand_mw: float
    renewable_pct: float
    # Generation
    coal: float
    gas: float
    hydro: float
    wind: float
    solar: float
    total_generation: float
    semischeduled_generation: float
    # Weather
    avg_temp: float
    avg_humidity: float
    avg_wind_speed: float
    solar_radiation: float
    cloud_cover: float
    is_peak: float
    # Price lags
    price_lag1: float
    price_lag2: float
    price_lag24: float
    price_lag168: float
    # Demand lags
    demand_mw_lag1: float
    demand_mw_lag2: float
    demand_mw_lag24: float
    demand_mw_lag48: float
    demand_mw_lag168: float
    # Renewable lags
    renewable_pct_lag1: float
    renewable_pct_lag24: float
    # Weather lags
    avg_temp_lag1: float
    avg_temp_lag24: float
    solar_radiation_lag1: float
    solar_radiation_lag24: float
    wind_lag1: float
    wind_lag24: float
    solar_lag1: float
    solar_lag24: float
    coal_lag1: float
    gas_lag1: float
    total_generation_lag1: float
    # Spike history
    spike_lag1: float
    spike_lag2: float
    spike_roll4_sum: float
    spike_roll24_sum: float
    # Rolling stats
    price_roll4_mean: float
    price_roll4_std: float
    price_roll24_mean: float
    price_roll24_std: float
    demand_roll4_mean: float
    demand_roll4_std: float
    demand_roll24_mean: float
    demand_roll24_std: float
    temp_roll4_mean: float
    temp_roll24_mean: float
    solar_roll24_mean: float

def price_features(d: dict) -> pd.DataFrame:
    f = {}
    f['hour_sin']  = np.sin(2 * np.pi * d['hour_of_day'] / 24)
    f['hour_cos']  = np.cos(2 * np.pi * d['hour_of_day'] / 24)
    f['dow_sin']   = np.sin(2 * np.pi * d['day_of_week'] / 7)
    f['dow_cos']   = np.cos(2 * np.pi * d['day_of_week'] / 7)
    f['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    f['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)

    for col in ['demand_mw','renewable_pct','avg_temp','avg_humidity',
                'avg_wind_speed','solar_radiation','cloud_cover','is_peak',
                'coal','gas','hydro','wind','solar',
                'total_generation','semischeduled_generation',
                'price_lag1','price_lag2','price_lag24','price_lag168',
                'demand_mw_lag1','demand_mw_lag24',
                'renewable_pct_lag1','renewable_pct_lag24',
                'avg_temp_lag1','avg_temp_lag24',
                'solar_radiation_lag1','solar_radiation_lag24',
                'wind_lag1','wind_lag24','solar_lag1','solar_lag24',
                'coal_lag1','gas_lag1','total_generation_lag1',
                'price_roll4_mean','price_roll4_std',
                'price_roll24_mean','price_roll24_std',
                'demand_roll4_mean','demand_roll24_mean',
                'temp_roll24_mean','solar_roll24_mean']:
        f[col] = d[col]

    f['price_range']         = d['price_max'] - d['price_min']
    f['temp_x_demand']       = d['avg_temp'] * d['demand_mw']
    f['renewable_x_demand']  = d['renewable_pct'] * d['demand_mw']
    f['temp_x_peak']         = d['avg_temp'] * d['is_peak']
    f['solar_x_cloud']       = d['solar_radiation'] * (1 - d['cloud_cover'] / 100)
    f['demand_dev24']        = d['demand_mw'] - d['demand_roll24_mean']
    f['price_momentum']      = d['price_lag1'] - d['price_lag24']

    return pd.DataFrame([f])[[
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'demand_mw', 'renewable_pct', 'avg_temp', 'avg_humidity', 'avg_wind_speed',
        'solar_radiation', 'cloud_cover', 'is_peak',
        'coal', 'gas', 'hydro', 'wind', 'solar',
        'total_generation', 'semischeduled_generation',
        'price_range',
        'price_lag1', 'price_lag2', 'price_lag24', 'price_lag168',
        'demand_mw_lag1', 'demand_mw_lag24',
        'renewable_pct_lag1', 'renewable_pct_lag24',
        'avg_temp_lag1', 'avg_temp_lag24',
        'solar_radiation_lag1', 'solar_radiation_lag24',
        'wind_lag1', 'wind_lag24', 'solar_lag1', 'solar_lag24',
        'coal_lag1', 'gas_lag1', 'total_generation_lag1',
        'price_roll4_mean', 'price_roll4_std',
        'price_roll24_mean', 'price_roll24_std',
        'demand_roll4_mean', 'demand_roll24_mean',
        'temp_roll24_mean', 'solar_roll24_mean',
        'temp_x_demand', 'renewable_x_demand', 'temp_x_peak', 'solar_x_cloud',
        'demand_dev24', 'price_momentum',
    ]]

def demand_features(d: dict) -> pd.DataFrame:
    f = {}
    f['hour_sin']  = np.sin(2 * np.pi * d['hour_of_day'] / 24)
    f['hour_cos']  = np.cos(2 * np.pi * d['hour_of_day'] / 24)
    f['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)

    for col in ['demand_mw_lag1','demand_mw_lag2','demand_mw_lag24',
                'demand_mw_lag48','demand_mw_lag168',
                'demand_roll4_mean','demand_roll24_mean',
                'solar_radiation_lag1','is_peak']:
        f[col] = d[col]

    f['demand_momentum']    = d['demand_mw_lag1'] - d['demand_mw_lag24']
    f['temp_x_demand_lag1'] = d['avg_temp'] * d['demand_mw_lag1']
    f['solar_x_cloud']      = d['solar_radiation'] * (1 - d['cloud_cover'] / 100)

    return pd.DataFrame([f])[[
        'hour_sin', 'hour_cos', 'month_cos',
        'demand_mw_lag1', 'demand_mw_lag2', 'demand_mw_lag24',
        'demand_mw_lag48', 'demand_mw_lag168',
        'demand_roll4_mean', 'demand_roll24_mean',
        'demand_momentum', 'temp_x_demand_lag1',
        'solar_radiation_lag1', 'is_peak', 'solar_x_cloud',
    ]]

def spike_features(d: dict) -> pd.DataFrame:
    f = {}
    f['hour_sin']  = np.sin(2 * np.pi * d['hour_of_day'] / 24)
    f['hour_cos']  = np.cos(2 * np.pi * d['hour_of_day'] / 24)
    f['dow_sin']   = np.sin(2 * np.pi * d['day_of_week'] / 7)
    f['dow_cos']   = np.cos(2 * np.pi * d['day_of_week'] / 7)
    f['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    f['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)

    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,
                  6:2,7:2,8:2,9:3,10:3,11:3}
    f['season'] = season_map[d['month']]

    for col in ['demand_mw','renewable_pct','avg_temp','avg_humidity',
                'avg_wind_speed','solar_radiation','cloud_cover','is_peak',
                'coal','gas','wind','solar','total_generation',
                'semischeduled_generation',
                'price_lag1','price_lag2','price_lag24',
                'demand_mw_lag1','demand_mw_lag24',
                'renewable_pct_lag1','renewable_pct_lag24',
                'avg_temp_lag1','avg_temp_lag24',
                'solar_radiation_lag1','solar_radiation_lag24',
                'wind_lag1','coal_lag1','gas_lag1',
                'spike_lag1','spike_lag2',
                'spike_roll4_sum','spike_roll24_sum',
                'price_roll4_mean','price_roll4_std',
                'price_roll24_mean','price_roll24_std',
                'demand_roll4_mean','demand_roll24_mean']:
        f[col] = d[col]

    f['price_range']         = d['price_max'] - d['price_min']
    f['temp_x_demand']       = d['avg_temp'] * d['demand_mw']
    f['renewable_x_demand']  = d['renewable_pct'] * d['demand_mw']
    f['temp_x_peak']         = d['avg_temp'] * d['is_peak']
    f['demand_dev24']        = d['demand_mw'] - d['demand_roll24_mean']
    f['price_momentum']      = d['price_lag1'] - d['price_lag24']
    f['supply_squeeze']      = d['total_generation'] / max(d['demand_mw'], 1)
    f['heat_x_low_wind']     = d['avg_temp'] * (1 / (d['wind_lag1'] + 1))
    f['heat_x_low_solar']    = d['avg_temp'] * (1 / (d['solar_radiation_lag1'] + 1))
    f['demand_supply_gap']   = d['demand_mw_lag1'] - d['total_generation']
    f['net_load']            = d['demand_mw'] - (d['wind'] + d['solar'])
    f['net_load_lag1']       = d['demand_mw_lag1'] - (d['wind_lag1'] + d['solar_lag1'])
    f['net_load_roll4']      = d['demand_roll4_mean']
    f['net_load_stress']     = float(f['net_load'] > NET_Q90)

    s = f['season']
    f['summer_heatwave']      = float(s == 0 and d['avg_temp'] > 25)
    f['summer_low_renewable'] = float(s == 0 and d['renewable_pct_lag1'] < 30)
    f['summer_stress']        = float(s == 0 and d['avg_temp'] > 25 and d['renewable_pct_lag1'] < 30)
    f['winter_low_wind']      = float(s == 2 and d['wind_lag1'] < WIND_Q25)
    f['high_pressure_proxy']  = float(d['avg_wind_speed'] < 10 and d['avg_temp'] > 20)
    f['low_renewable_stress'] = float(d['renewable_pct_lag1'] < 15 and d['demand_mw_lag1'] > 7000)

    return pd.DataFrame([f])[[
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'season',
        'demand_mw', 'renewable_pct', 'avg_temp', 'avg_humidity', 'avg_wind_speed',
        'solar_radiation', 'cloud_cover', 'is_peak',
        'coal', 'gas', 'wind', 'solar', 'total_generation', 'semischeduled_generation',
        'price_range',
        'price_lag1', 'price_lag2', 'price_lag24',
        'demand_mw_lag1', 'demand_mw_lag24',
        'renewable_pct_lag1', 'renewable_pct_lag24',
        'avg_temp_lag1', 'avg_temp_lag24',
        'solar_radiation_lag1', 'solar_radiation_lag24',
        'wind_lag1', 'coal_lag1', 'gas_lag1',
        'price_roll4_mean', 'price_roll4_std',
        'price_roll24_mean', 'price_roll24_std',
        'demand_roll4_mean', 'demand_roll24_mean',
        'spike_lag1', 'spike_lag2',
        'spike_roll4_sum', 'spike_roll24_sum',
        'temp_x_demand', 'renewable_x_demand', 'temp_x_peak',
        'demand_dev24', 'price_momentum',
        'summer_heatwave', 'summer_low_renewable', 'summer_stress',
        'winter_low_wind', 'high_pressure_proxy',
        'supply_squeeze', 'heat_x_low_wind', 'heat_x_low_solar',
        'demand_supply_gap', 'low_renewable_stress',
        'net_load', 'net_load_lag1', 'net_load_roll4', 'net_load_stress',
    ]]

@app.get("/")
def root():
    return {"message": "Melbourne Energy Prediction API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/features")
def features():
    return {"required_fields": list(RawInput.model_fields.keys())}

@app.post("/predict")
def predict(data: RawInput):
    d = data.dict()

    X_price  = price_features(d)
    X_demand = demand_features(d)
    X_spike  = spike_features(d)

    price      = float(np.expm1(price_model.predict(X_price)[0]) - PRICE_SHIFT)
    demand     = float(demand_model.predict(X_demand)[0])
    spike_prob = float(spike_model.predict_proba(X_spike)[0][1])
    spike_alert = int(spike_prob >= 0.7)

    return {
        "predicted_price"   : round(price, 2),
        "predicted_demand"  : round(demand, 0),
        "spike_probability" : round(spike_prob, 4),
        "spike_alert"       : spike_alert,
        "spike_alert_label" : "SPIKE RISK" if spike_alert == 1 else "NORMAL",
    }
