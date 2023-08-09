import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle

# Charger le modèle entrainé
model = joblib.load('model_opti_1.pkl')

# Fonction de prédiction et de probabilité
def predict_fake_bill(features):
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return prediction, proba

# Fonction pour créer un graphique de jauge avec matplotlib
def gauge_chart(probability):
    fig, ax = plt.subplots()
    ax.axis('off')
    ang_range = 180
    start_ang = 90 + ang_range / 2
    end_ang = 90 - ang_range / 2
    radius = 0.5
    width = 0.2
    colors = ['red', 'green']
    wedges = []
    wedges.append(Wedge((0,0), radius, end_ang, start_ang, width=width))
    wedges.append(Wedge((0,0), radius, end_ang, end_ang + probability * ang_range, width=width))
    for i in range(len(wedges)):
        ax.add_artist(wedges[i])
        wedges[i].set_facecolor(colors[i])
        wedges[i].set_edgecolor('white')
        wedges[i].set_linewidth(2)
    ax.text(0, 0-radius/2-0.1, f'{probability:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=16)
    return fig

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
        if prediction == True:
            st.write('Le billet est authentique.')
            probability = proba[1]
        else:
            st.write('Le billet est un faux.')
            probability = proba[0]
        st.write(f'Probabilité d\'obtention d\'un vrai billet : {proba[1]}')
        st.write(f'Probabilité d\'obtention d\'un faux billet : {proba[0]}')
        
        # Ajout de la jauge pour afficher la probabilité d'obtention du billet
        fig = gauge_chart
        fig = gauge_chart(probability)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
