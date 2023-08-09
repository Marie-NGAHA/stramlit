import streamlit as st
import joblib
import pandas as pd

# Charger le modèle entrainé
model = joblib.load('model_opti_1.pkl')

# Fonction de prédiction et de probabilité
def predict_fake_bill(features):
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return prediction, proba

# Créer votre interface Streamlit
def main():
    st.title('Détection de faux billet')
    st.write('Veuillez entrer les caractéristiques du billet à analyser :')

    # Entrée des caractéristiques du billet
    diagonal = st.number_input('Diagonal', min_value=0.0)
    height_left = st.number_input('Hauteur gauche', min_value=0.0)
    height_right = st.number_input('Hauteur droite', min_value=0.0)
    margin_low = st.number_input('Marge inférieure', min_value=0.0)
    margin_up = st.number_input('Marge supérieure', min_value=0.0)
    length = st.number_input('Longueur', min_value=0.0)

    # Bouton pour prédire
    if st.button('Prédire'):
        features = [diagonal, height_left, height_right, margin_low, margin_up, length]
        prediction, proba = predict_fake_bill(features)
        if prediction == 0:
            st.write('Le billet est authentique.')
        else:
            st.write('Le billet est un faux.')
        st.write(f'Probabilité d\'obtention d\'un vrai billet : {proba[0]:.2f}')
        st.write(f'Probabilité d\'obtention d\'un faux billet : {proba[1]:.2f}')

if __name__ == '__main__':
    main()
