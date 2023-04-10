#from ast import With
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
from lightgbm import LGBMClassifier
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle
import math
import base64
from flask import jsonify
import json
app= Flask(__name__)

df_test = pickle.load(open('X_test_low.pickle', 'rb'))
df_train = pickle.load(open('X_train_low.pickle', 'rb'))
model = pickle.load(open('model.pickle', 'rb'))
knn = pickle.load(open('knn_low.pickle', 'rb'))
app_train = pickle.load(open('app_train_low.pickle', 'rb'))
app_test = pickle.load(open('app_test_low.pickle', 'rb'))
id_client = df_test["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

@app.route("/load_id_client", methods=["GET"])
def load_data():
    id_client = df_test["SK_ID_CURR"][:50].values
    id_client = pd.DataFrame(id_client)
    return id_client.to_json(orient='values')

@app.route("/infos_client", methods=["GET"])
def infos_client():
    id = request.args.get("id_client")
    data_client = app_test[app_test["SK_ID_CURR"] == int(id)]
    dict_infos = {
       #"status_famille": data_client["NAME_FAMILY_STATUS"].item(),
       "nb_enfant": data_client["CNT_CHILDREN"].item(),
       "age": int(data_client["DAYS_BIRTH"].values / -365),
       "revenus": data_client["AMT_INCOME_TOTAL"].item(),
       "montant_credit": data_client["AMT_CREDIT"].item(),
       "annuites": data_client["AMT_ANNUITY"].item(),
       "montant_bien": data_client["AMT_GOODS_PRICE"].item(),
       "metier"      : data_client["OCCUPATION_TYPE"].item()
    }
    # dict_infos = {
    #    "status_famille": data_client["NAME_FAMILY_STATUS"].item(),
    #    "nb_enfant": data_client["CNT_CHILDREN"].item(),
    #    "age": int(data_client["DAYS_BIRTH"].values / -365),
    #    "revenus": data_client["AMT_INCOME_TOTAL"].item(),
    #    "montant_credit": data_client["AMT_CREDIT"].item(),
    #    "annuites": data_client["AMT_ANNUITY"].item(),
    #    "montant_bien": data_client["AMT_GOODS_PRICE"].item()
    # }
    #response = json.dumps(dict_infos)  # Convertir le dictionnaire en JSON
    return jsonify(dict_infos)

@app.route('/predict', methods = ['GET'])
def predict():
    id = request.args.get("id_client")
    liste_clients = list(df_test['SK_ID_CURR'].unique())
    probability_default_payment = 0
    seuil = 0.5
    id = int(id)
    if id not in liste_clients:
        prediction="Ce client n'est pas répertorié"
    else :
        X = df_test[df_test['SK_ID_CURR'] == id]
        X.drop('SK_ID_CURR', axis=1, inplace=True) 
        print(X)
        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Prêt Accordé"
        else:
            prediction = "Prêt Non Accordé"
    return jsonify({"prediction": prediction, "score": probability_default_payment.tolist()})


@app.route("/get_client_features", methods=["GET"])
def get_client_features():
    id = request.args.get("id_client")
    client_data = df_test[df_test["SK_ID_CURR"] == int(id)]
    client_data['index'] = client_data.index
    print(client_data)
    client_data.drop('SK_ID_CURR', axis=1, inplace=True) 
    #print(client_data)
    return jsonify(client_data.to_dict(orient='records')[0])
    #return jsonify(client_data.to_dict(orient='records'))




# @app.route("/load_voisins", methods=["GET"])
# def load_voisins():
#     id = request.args.get("id_client")
#     interpretable_important_data = ['SK_ID_CURR',
#                                     'PAYMENT_RATE',
#                                     'AMT_ANNUITY',
#                                     'DAYS_BIRTH',
#                                     'DAYS_EMPLOYED',
#                                     'ANNUITY_INCOME_PERC']
#     data_client = df_test[df_test["SK_ID_CURR"] == int(id)][interpretable_important_data].values
#     distances, indices = knn.kneighbors(data_client)
#     df_train_selected = df_train[interpretable_important_data]  
#     df_voisins = df_train_selected.iloc[indices[0], :]  
#     return jsonify(df_voisins.to_dict(orient='records'))

@app.route("/load_voisins", methods=["GET"])
#id = 100005
def load_voisins():
    id = request.args.get("id_client")
    # interpretable_important_data = [    'PAYMENT_RATE',
    #                                 'AMT_ANNUITY',
    #                                 'DAYS_BIRTH',
    #                                 'DAYS_EMPLOYED',
    #                                 'ANNUITY_INCOME_PERC']
    interpretable_important_data = ['EXT_SOURCE_2',
                                    'EXT_SOURCE_3',
                                    'EXT_SOURCE_1',
                                    'CODE_GENDER',
                                    'PAYMENT_RATE',
                                    'INSTAL_DPD_MEAN',
                                    'INSTAL_AMT_PAYMENT_SUM',
                                    'APPROVED_CNT_PAYMENT_MEAN',
                                    'AMT_ANNUITY',
                                    'DAYS_BIRTH',
                                    'DAYS_EMPLOYED',
                                    'NAME_EDUCATION_TYPE_Highereducation',
                                    'FLAG_OWN_CAR',
                                    'DAYS_EMPLOYED_PERC',
                                    'NAME_FAMILY_STATUS_Married']
    #selection client
    data_client = df_test[df_test["SK_ID_CURR"] == int(id)][interpretable_important_data]
    distances, indices = knn.kneighbors(data_client)
    X_train_selected = df_train[interpretable_important_data]
    # selections des plus proches voisins
    df_voisins = X_train_selected.iloc[indices[0], :]
    # Selection des colonnes à afficher
    affichage_colonne = [    
                            'SK_ID_CURR',
                            'AMT_ANNUITY',
                             'DAYS_BIRTH',
                             'DAYS_EMPLOYED',
                             'NAME_EDUCATION_TYPE',
                             'OCCUPATION_TYPE',

                         ]
    df_voisins = app_train.loc[df_voisins.index][affichage_colonne]
    # traitement champ Day employed
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    def days_to_duration(days):
        now = datetime.now()
        target_date = now + timedelta(days=days)
        delta = relativedelta(target_date, now)
        return f"{abs(delta.years)}y {abs(delta.months)}m {abs(delta.days)}d"
    df_voisins['DAYS_EMPLOYED'] = df_voisins['DAYS_EMPLOYED'].apply(days_to_duration)
    # traitement de Day birth
    df_voisins['DAYS_BIRTH'] = df_voisins['DAYS_BIRTH'].apply(lambda x: int(x / -365))
    return jsonify(df_voisins.to_dict(orient='records'))




if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=True)
