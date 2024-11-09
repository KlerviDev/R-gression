import numpy as np
import pandas as pd  # Utilisé pour la manipulation de données
import joblib  # Utilisé pour charger le modèle sauvegardé
from flask import Flask, request, jsonify  # Flask est un micro-framework pour les applications web


# Charger le modèle de forêt aléatoire depuis le disque
modele = joblib.load('random_forest_model.pkl')

# Création de l'instance de l'application Flask
app = Flask(__name__)

# Définition de la route racine qui retourne un message de bienvenue
@app.route("/", methods=["GET"])
def accueil():
    """ Endpoint racine qui fournit un message de bienvenue. """
    return jsonify({"message": "Bienvenue sur l'API de prédiction de prix"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Récupérer les données d'entrée en JSON
    
    # Préparer les données pour la prédiction
    features = np.array(data['features']).reshape(1, -1)

    # Prédire le résultat
    prediction = modele.predict(features)
    
    # Renvoyer le résultat sous forme de JSON
    return jsonify({'prediction sale': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

#Vous pouvez tester le modèle avec Postman !!!   
