import streamlit as st
import pandas as pd
import numpy as np
import requests 
import json
import os

# Configuration
API_URL = "http://localhost:8000/signal"
TIME_STEPS = 30

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_processed.csv')
BACKTEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_backtest.csv')

# Streamlit App Configuration
st.set_page_config(page_title="Quant Trading Dashboard", layout="wide")
st.title("Quant Trading Sequential Prediction Dashboard")
st.markdown("""This dashboard allows you to visualize cryptocurrency data and obtain trading signals using a pre-trained LSTM model via a FastAPI backend.""")

# Function to load backtest data
def load_backtest_data():
    if not os.path.exists(BACKTEST_DATA_PATH):
         st.error(f"Erreur Fichier Introuvable: Le fichier de backtest ({BACKTEST_DATA_PATH}) n'existe pas. Veuillez l'exécuter et le sauvegarder depuis le notebook (Étape 22).")
         return None
    try:
        df = pd.read_csv(BACKTEST_DATA_PATH, index_col='Date', parse_dates=True)
        required_cols = ['Cumulative_Market_Return', 'Cumulative_Strategy_Return', 'Sharpe_Ratio', 'Max_Drawdown']
        if not all(col in df.columns for col in required_cols):
             st.error(f"Erreur Colonnes Manquantes: Le fichier {BACKTEST_DATA_PATH} doit contenir les colonnes {required_cols}. Vérifiez l'Étape 22 du notebook.")
             return None
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier de backtest ({BACKTEST_DATA_PATH}): {e}")
        return None


def get_sample_sequence(processed_data_path):
    """Charge les données traitées et retourne la séquence la plus récente."""
    if not os.path.exists(processed_data_path):
         st.error(f"Erreur Fichier Introuvable: Le fichier de données traitées ({processed_data_path}) n'existe pas. Veuillez l'exécuter et le sauvegarder depuis le notebook (Étape 3).")
         return None
    try:
        df = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)
        if len(df) < TIME_STEPS:
            st.error(f"Pas assez de données dans {processed_data_path} pour créer une séquence de {TIME_STEPS} jours.")
            return None
        sample = df.tail(TIME_STEPS)
        
        features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_100', 'RSI_14', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        
        if not all(col in sample.columns for col in features_list):
             missing = [col for col in features_list if col not in sample.columns]
             st.error(f"Erreur Colonnes Manquantes dans {processed_data_path}: {missing}. Vérifiez l'Étape 3 du notebook.")
             return None
         
        
        return sample[features_list].to_dict('records')
    except Exception as e:
        st.error(f"Erreur de chargement des données de test ({processed_data_path}): {e}")
        return None


def display_results():
    df_backtest = load_backtest_data()
    col1, col2 = st.columns(2)
    with col1:
         st.subheader("Performance Historique (Backtest)")
         if df_backtest is not None:
             st.line_chart(df_backtest[['Cumulative_Market_Return', 'Cumulative_Strategy_Return']])
             sharpe = df_backtest['Sharpe_Ratio'].iloc[0] 
             mdd = df_backtest['Max_Drawdown'].iloc[0]
             sharpe_str = f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
             mdd_str = f"{mdd:.2f}%" if pd.notna(mdd) else "N/A"
             st.markdown(f"**Ratio de Sharpe : {sharpe_str}** | **Max Drawdown : {mdd_str}**")
         else:
             st.warning("Impossible d'afficher le graphique de backtest.")
             
    with col2: 
         st.subheader("Test en Temps Réel (via API)")

         if st.button("Obtenir le Signal sur la dernière séquence"):
             with st.spinner('Appel de l’API et inférence du modèle LSTM...'):
                 sample_data = get_sample_sequence(PROCESSED_DATA_PATH)

                 if sample_data:
                     try:
                         response = requests.post(API_URL, json={"sequence": sample_data}, timeout=15)
                         response.raise_for_status()

                         result = response.json()
                         interpretation = result.get('interpretation', 'N/A')
                         proba = result.get('prediction_probability', 'N/A')
                         threshold = result.get('threshold_used', 'N/A')

                         st.success(f"Signal reçu : **{interpretation}**")
                         st.metric(label="Probabilité d'Hausse (Classe 1)", value=f"{proba:.4f}" if isinstance(proba, float) else proba, delta=f"Seuil: {threshold}" if threshold != 'N/A' else None)

                     except requests.exceptions.ConnectionError:
                         st.error(f"Erreur de Connexion: L'API Docker sur {API_URL} ne répond pas. Est-elle lancée ?")
                     except requests.exceptions.Timeout:
                         st.error("Erreur Timeout: L'API a mis trop de temps à répondre.")
                     except requests.exceptions.RequestException as e:
                         st.error(f"Erreur API ({e.response.status_code if e.response else 'N/A'}): {e.response.json().get('detail', 'Réponse inconnue') if e.response else str(e)}")
                     except Exception as e:
                         st.error(f"Une erreur inattendue est survenue lors de l'appel API: {e}")
                 else:
                      st.error("Impossible de charger les données pour l'appel API.")

if __name__ == "__main__":
    display_results()