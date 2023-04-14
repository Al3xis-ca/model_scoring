import requests
import streamlit as st
import pandas as pd
import pickle
import warnings
import shap
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
from PIL import Image

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Loan Scoring APP', layout="wide")
URL_API = "http://127.0.0.1:5000/"

df_test = pickle.load(open('X_test_low.pickle', 'rb'))                  
shap_values = pickle.load(open('shap_values.pickle', 'rb'))             

sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
'''

st.markdown(sysmenu, unsafe_allow_html=True)
df_dashboard = pickle.load(open('X_test_low.pickle', 'rb'))
# model = pickle.load(open('LGBM.pickle', 'rb')).best_estimator_
model = pickle.load(open('model.pickle', 'rb'))

df_test.drop(columns=['SK_ID_CURR'], inplace=True)                  
shap.summary_plot(shap_values, df_test, plot_type="bar")
plt.savefig('features_importance.png', dpi=1000)

def main():
    global id_client
    st.sidebar.image('logo.png')
    lst_id = load_id_client()
    id_client = st.sidebar.selectbox("ID Client", lst_id)
    prediction = load_prediction()
    #st.write(prediction)
    
    if prediction['prediction'] == "Prêt Accordé":
        st.sidebar.markdown(
        "<div style='background-color: #90ee90; padding: 10px; border-radius: 5px;  font-weight: bold; font-size: 20px; text-align: center;'>Prêt Accordé</div></div>",
        unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='background-color: #ff7f7f; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 20px; text-align: center; '>Prêt Refusé</div>",
            unsafe_allow_html=True,
        )

    
    
    score = prediction['score']
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score[0] * 100,
    title={'text': "Probabilité de faillite", 'font': {'size': 20}},  # Augmenter la taille du titre
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "red"},  # Change la couleur de la partie mobile de la jauge
        'steps': [
            {'range': [0, 25], 'color': "lightgray"},
            {'range': [25, 50], 'color': "gray"},
            {'range': [50, 75], 'color': "lightsalmon"},  # Change la couleur de fond
            {'range': [75, 100], 'color': "salmon"}],  # Change la couleur de fond
        'threshold': {
            'line': {'color': "red", 'width': 3},
            'thickness': 0.75,
            'value': score[0] * 100}}))
    
           

    fig.update_layout(height=250, width=350,
                    font={'color': 'black', 'family': 'Sofia Pro', 'size': 20},
                    margin=dict(l=50, r=50, b=50, t=50, pad=50))
    fig.update_traces(number_font_size=50)

    # Centrer la figure dans Streamlit
    st.write('<style>div.row-widget.stCard {margin: 0 auto;}</style>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'responsive': False})
        
           
    st.write("### Informations Client")
    st.write(infos_client())
    
    st.write("### Informations Clients même profile")
    st.write(load_voisins())
        
    client_features = get_client_features()
    # Test retour features client
    st.write(client_features)
    
       
    tab1, tab2, tab3 = st.tabs(["Client score", "Importance features", "Informations client, clients de même profile"])

    with tab1:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            #st.image(features_client(), use_column_width=True)
            idx = int(client_features['index'])
            explainer = TreeExplainer(model['model'])
            observations = model['std'].transform(df_test)
            sv = explainer(observations)
            exp = Explanation(sv.values[:,:,1], 
                          sv.base_values[:,1], 
                          data=observations, 
                          feature_names=df_test.columns)
            plt.figure(figsize=(100, 100))
            plt.rcParams.update({'font.size': 6})
            waterfall(exp[idx])
            plt.subplots_adjust(left=0.6)
            plt.savefig('features_client.png')
            st.image('features_client.png', use_column_width=True)
    with tab2:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image('features_importance.png', use_column_width=True)
    
    with tab3:
        
        st.image('occupation_type.png', use_column_width=True)
        st.write("### Client")
        st.write(infos_client()["Metier"])
        st.write("### Clients même profile")
        st.write(load_voisins()['Metier'])
        
        st.image('education_type.png', use_column_width=True)
        st.write("### Client")
        st.write(infos_client()["Education"])          
        st.write("### Clients même profile")
        st.write(load_voisins()['Education'])
          
         
@st.cache_data()
def load_logo():
    # Construction de la sidebar
    # Chargement du logo
    logo = Image.open("logo.png") 
    return logo

def get_client_features():
    response = requests.get(URL_API + "get_client_features", params={"id_client": id_client})
    client_features = json.loads(response.content.decode("utf-8"))
    client_features = pd.DataFrame.from_dict(client_features, orient='index').T
    return client_features
    


def infos_client():
    # Requête permettant de récupérer les informations du client sélectionné
    infos_client = requests.get(URL_API + "infos_client", params={"id_client": str(id_client)})
    # On transforme la réponse en dictionnaire python
    infos_client = json.loads(infos_client.content.decode("utf-8"))
    # On transforme le dictionnaire en dataframe
    #infos_client = pd.DataFrame.from_dict(infos_client).T
    infos_client = pd.DataFrame(infos_client, index=[int(id_client)])
    #infos_client = infos_client.reset_index().drop(columns=['index'])
    return infos_client

 
def load_prediction():
    # Requête permettant de récupérer la prédiction
    # de faillite du client sélectionné
    prediction = requests.get(URL_API + "predict", params={"id_client":id_client})
    prediction = prediction.json()
    return prediction

def load_voisins():
    # Requête permettant de récupérer les 5 dossiers
    voisins = requests.get(URL_API + "load_voisins", params={"id_client":id_client})
    voisins = json.loads(voisins.content.decode("utf-8"))
    #print(voisins)
    # On transforme le dictionnaire en dataframe
    voisins = pd.DataFrame.from_dict(voisins)
    #voisins = pd.DataFrame(voisins)
    
    return voisins
      
@st.cache_data()
def load_id_client():
    # Requête permettant de récupérer la liste des ID clients
    data_json = requests.get(URL_API + "load_id_client")
    data = data_json.json()
    # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i[0])
    return lst_id
 

if __name__ == "__main__":
    main()
