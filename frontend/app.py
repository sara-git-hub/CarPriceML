import streamlit as st
import requests
import os
import json
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="CarPrice ML - Estimation de prix",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (peut être configurée via variable d'environnement)
API_URL = os.getenv("API_URL")

# Titre principal
st.title("🚗 CarPrice ML - Estimation du prix des voitures d'occasion")
st.markdown("### Prédiction intelligente du prix en dirhams marocains (MAD)")

# Sidebar avec informations
with st.sidebar:
    st.header("ℹ️ À propos")
    st.info(
        "Cette application utilise le Machine Learning pour estimer "
        "le prix d'une voiture d'occasion au Maroc en fonction de ses caractéristiques."
    )
    
    st.header("📊 Caractéristiques du modèle")
    st.markdown("""
    - **Algorithme**: Random Forest Regressor
    - **Features**: année, Puissance maximale (chevaux),Couple moteur (Nm),Cylindrée du moteur (cm³), âge
    - **Devise**: Dirham Marocain (MAD)
    """)
    
    # Vérifier la santé de l'API
    st.header("🔌 État du service")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API connectée")
        else:
            st.error("❌ API non disponible")
    except Exception as e:
        st.error(f"❌ Erreur de connexion: {str(e)}")

# Section principale
st.markdown("---")
st.header("📝 Saisissez les caractéristiques du véhicule")

# Créer deux colonnes pour le formulaire
col1, col2 = st.columns(2)

with col1:
    max_power_bhp = st.number_input(
        "Puissance maximale (chevaux)",
        min_value=10,
        max_value=150,
        value=74,
        step=1,
        help="Puissance maximale (chevaux)"
    )
    
    year = st.slider(
        "📅 Année de fabrication",
        min_value=1990,
        max_value=2025,
        value=2015,
        help="Année de mise en circulation du véhicule"
    )

with col2:
    torque_nm = st.number_input(
        "Couple moteur (Nm)",
        min_value=0,
        max_value=500,
        value=190,
        step=1,
        help="Couple moteur (Nm)"
    )
    
    engine_cc = st.number_input(
        "Cylindrée du moteur (cm³)",
        min_value=0,
        max_value=40000,
        value=1248,
        step=1,
        help="Cylindrée du moteur (cm³)"
    )

# Afficher les informations calculées
st.markdown("---")
st.subheader("📋 Résumé des informations")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric("Puissance maximale ", max_power_bhp)

with col_info2:
    vehicle_age = 2025 - year
    st.metric("Âge du véhicule", f"{vehicle_age} ans")

with col_info3:
    st.metric("Couple moteur", torque_nm)

with col_info4:
    st.metric("Cylindrée du moteur", engine_cc)

# Bouton de prédiction
st.markdown("---")
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])

with col_button2:
    predict_button = st.button(
        "🔮 Estimer le prix",
        use_container_width=True,
        type="primary"
    )

# Effectuer la prédiction
if predict_button:
    with st.spinner("🔄 Calcul du prix en cours..."):
        try:
            # Préparer les données
            car_data = {
                "year": year,
                "max_power_bhp": max_power_bhp,
                "torque_nm": torque_nm,
                "engine_cc": engine_cc
            }
            
            # Envoyer la requête à l'API
            response = requests.post(
                f"{API_URL}/predict",
                json=car_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result["predicted_price"]
                
                # Afficher le résultat avec style
                st.markdown("---")
                st.success("✅ Estimation réussie !")
                
                # Affichage du prix en grand
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1 style='color: #1f77b4; font-size: 3em; margin: 0;'>{predicted_price:,.2f} MAD</h1>
                        <p style='font-size: 1.2em; color: #666;'>Prix estimé du véhicule</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Afficher les détails en colonnes
                st.markdown("### 📊 Détails de l'estimation")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.info(f"""
                    **Véhicule analysé:**
                    - Année: {year}
                    - Puissance maximale : {max_power_bhp}
                    - Âge: {vehicle_age} ans
                    """)
                
                with detail_col2:
                    st.info(f"""
                    **Caractéristiques:**
                    - Couple moteur: {torque_nm}
                    - Cylindrée du moteur: {engine_cc}
                    """)
                
                # Conseils basés sur le prix
                st.markdown("### 💡 Recommandations")
                
                if predicted_price > 200000:
                    st.warning("⚠️ Prix élevé - Vérifiez l'état général et l'historique du véhicule")
                elif predicted_price < 50000:
                    st.info("ℹ️ Prix abordable - Assurez-vous de l'état mécanique et de l'entretien")
                else:
                    st.success("✅ Prix dans la moyenne du marché")
                
            else:
                st.error(f"❌ Erreur {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("❌ Délai d'attente dépassé. Veuillez réessayer.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez que le service backend est actif.")
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {str(e)}")
