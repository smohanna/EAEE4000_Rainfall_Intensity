import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_berif_dataset(n_samples=10000):
    """
    Gera um dataset sintético para o projeto B-ERIF.
    Simula dados de estações, clima e intensidade de chuva baseada em física + ruído.
    """
    
    np.random.seed(42) # Para resultados reproduzíveis

    print(f"Gerando {n_samples} amostras de dados climáticos sintéticos...")

    # --- 1. Dados Espaciais (Estações no Brasil) ---
    stations = {
        'BR_SP_001': {'lat': -23.55, 'lon': -46.63, 'elev': 760, 'region': 'Sudeste'}, # São Paulo
        'BR_AM_002': {'lat': -3.11,  'lon': -60.02, 'elev': 92,  'region': 'Norte'},   # Manaus
        'BR_RS_003': {'lat': -30.03, 'lon': -51.22, 'elev': 10,  'region': 'Sul'},     # Porto Alegre
        'BR_CE_004': {'lat': -3.71,  'lon': -38.54, 'elev': 21,  'region': 'Nordeste'},# Fortaleza
        'BR_GO_005': {'lat': -16.68, 'lon': -49.26, 'elev': 749, 'region': 'Centro'}   # Goiânia
    }
    
    station_ids = np.random.choice(list(stations.keys()), n_samples)
    
    # --- 2. Dados Temporais ---
    start_date = datetime(2010, 1, 1)
    date_list = [start_date + timedelta(days=np.random.randint(0, 5000)) for _ in range(n_samples)]
    
    # --- 3. Parâmetros da Curva IFD ---
    durations = [10, 30, 60, 120, 360, 720, 1440] # minutos
    duration_col = np.random.choice(durations, n_samples)
    
    returns = [2, 5, 10, 25, 50, 100] # anos
    return_col = np.random.choice(returns, n_samples)

    # --- 4. Dados Climáticos ---
    temp_col = []
    humidity_col = []
    pressure_col = []
    lat_col = []
    lon_col = []
    elev_col = []

    for i in range(n_samples):
        st = stations[station_ids[i]]
        dt = date_list[i]
        
        # Sazonalidade simples
        month_factor = 1 if (dt.month >= 10 or dt.month <= 3) else -1
        base_temp = 25 + (month_factor * 3) 
        
        # Temperatura, Umidade, Pressão com ruído
        temp = np.random.normal(base_temp, 3) 
        hum = np.clip(np.random.normal(80 - (temp/2), 10), 30, 100)
        press = np.random.normal(1013, 5)
        
        temp_col.append(round(temp, 1))
        humidity_col.append(round(hum, 1))
        pressure_col.append(round(press, 1))
        lat_col.append(st['lat'])
        lon_col.append(st['lon'])
        elev_col.append(st['elev'])

    # --- 5. Gerar TARGET: Intensidade (mm/h) ---
    # Equação baseada em física (IFD) + Fator Climático
    K, a, b, c = 500, 0.2, 10, 0.8
    
    intensity = (K * (return_col ** a)) / ((duration_col + b) ** c)
    
    # Ajuste Climático (tempestades convectivas com calor)
    climate_factor = 1 + (np.array(temp_col) - 20) * 0.02
    intensity = intensity * climate_factor
    
    # Ruído
    noise = np.random.normal(0, intensity * 0.1) 
    intensity_final = np.abs(intensity + noise)

    # --- 6. Montar o DataFrame ---
    df = pd.DataFrame({
        'station_id': station_ids,
        'datetime': date_list,
        'year': [d.year for d in date_list],
        'month': [d.month for d in date_list],
        'latitude': lat_col,
        'longitude': lon_col,
        'elevation_m': elev_col,
        'duration_min': duration_col,
        'return_period_years': return_col,
        'avg_temperature_c': temp_col,
        'relative_humidity': humidity_col,
        'atmospheric_pressure': pressure_col,
        'intensity_mm_h': np.round(intensity_final, 2)
    })
    
    df = df.sort_values(by=['station_id', 'datetime']).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df_berif = generate_berif_dataset()
    filename = 'berif_synthetic_data.csv'
    df_berif.to_csv(filename, index=False)
    print(f"\nArquivo '{filename}' gerado com sucesso!")
    print("Agora você pode prosseguir para o treinamento dos modelos.")