import pandas as pd
import numpy as np
import time

# Scikit-Learn (Substituindo XGBoost por GradientBoostingRegressor)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import GradientBoostingRegressor # <--- O Substituto do XGBoost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Configuração e Carregamento ---
print(">>> Carregando dataset...")
try:
    df = pd.read_csv('berif_synthetic_data.csv')
except FileNotFoundError:
    print("ERRO: Arquivo csv não encontrado. Rode o gerar_dataset.py primeiro.")
    exit()

features_num = ['duration_min', 'return_period_years', 'avg_temperature_c', 
                'relative_humidity', 'atmospheric_pressure', 'elevation_m']
features_cat = ['region'] if 'region' in df.columns else ['station_id'] 

X = df[features_num + features_cat]
y = df['intensity_mm_h']

# Divisão 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Pipeline de Pré-processamento ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ])

print(">>> Processando dados...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==============================================================================
# MODELO 1: Gradient Boosting (Equivalente ao XGBoost)
# ==============================================================================
print("\n" + "="*50)
print("TREINANDO MODELO 1: Gradient Boosting (GBM)")
print("="*50)
start_time = time.time()

gbm_model = GradientBoostingRegressor(random_state=42)

# Grid Simplificado
param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

print(">>> Otimizando hiperparâmetros...")
search = RandomizedSearchCV(gbm_model, param_grid, n_iter=6, cv=3, 
                           scoring='neg_root_mean_squared_error', verbose=1, random_state=42)
search.fit(X_train_processed, y_train)

best_gbm = search.best_estimator_
gbm_time = time.time() - start_time

print(f"Melhores parâmetros GBM: {search.best_params_}")
y_pred_gbm = best_gbm.predict(X_test_processed)

# ==============================================================================
# MODELO 2: Neural Network (MLP)
# ==============================================================================
print("\n" + "="*50)
print("TREINANDO MODELO 2: Rede Neural (MLP)")
print("="*50)
nn_start_time = time.time()

model_nn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    learning_rate_init=0.001, batch_size=32,
    max_iter=300, early_stopping=True,
    random_state=42, verbose=False
)

model_nn.fit(X_train_processed, y_train)

nn_time = time.time() - nn_start_time
y_pred_nn = model_nn.predict(X_test_processed)

# ==============================================================================
# COMPARAÇÃO
# ==============================================================================
def evaluate(y_true, y_pred, model_name, time_taken):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "RMSE (mm/h)": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 3),
        "Training Time (s)": round(time_taken, 1)
    }

results = [
    evaluate(y_test, y_pred_gbm, "Gradient Boosting (GBM)", gbm_time),
    evaluate(y_test, y_pred_nn, "Neural Network (MLP)", nn_time)
]

df_results = pd.DataFrame(results)

print("\n\nRESULTADOS FINAIS:")
print(df_results)
df_results.to_csv('final_results.csv', index=False)