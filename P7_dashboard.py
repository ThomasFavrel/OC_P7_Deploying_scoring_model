from json import load
import pandas as pd
import streamlit as st
import requests
import numpy as np
import mlflow
from p7_functions import mean_zero, mean_un, mean_data
import pickle
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
# data = pd.read_csv(r'Data/data_test_ohe.csv', index_col='SK_ID_CURR')
# data.drop(columns='Unnamed: 0', inplace=True)
# index_list = data.index.to_list()


# @st.cache
# def 
# model = mlflow.sklearn.load_model('mlflow_model_final_lgbm_test')
# data_cli = pd.DataFrame(data.loc[100002, :]).T.drop(columns='TARGET')
# pred = model.predict(data_cli)[0]

shap_values = pickle.load(open('shape_values.pickle', 'rb'))


data = pd.read_csv(r'Data/data_test_ohe.csv', index_col='SK_ID_CURR')
data.drop(columns='Unnamed: 0', inplace=True)
index_list = data.index.to_list()


data['ind'] = np.linspace(0, 306486, 306487)

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = data
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations' 

    st.title(
        'Prêt à dépenser'
    )

    CLIENT_INDEX = st.selectbox(
        'Please enter ID of the client',
        index_list
    )

    ind = int(data.loc[CLIENT_INDEX, 'ind'])
    model = mlflow.sklearn.load_model('mlflow_model_final222')
    data_cli = pd.DataFrame(data.loc[CLIENT_INDEX, :]).T.drop(columns=['TARGET', 'ind'])
    pred = model.predict(data_cli)[0]
    # predict_btn = st.button('Prédire')


    # if predict_btn:

    # data_cli = pd.DataFrame(data.loc[100002, :]).T.drop(columns='TARGET').fillna('').to_dict(orient='split')
    # data_cli = pd.DataFrame(data.loc[100002, :]).T.drop(columns='TARGET').to_dict(orient='split')
    # data_cli = pd.DataFrame(data.loc[CLIENT_INDEX, :]).T.drop(columns='TARGET')

    # pred = request_prediction(MLFLOW_URI, data_cli)[0]
    # pred = model.predict(data_cli)[0]
    pred = 100 - pred * 100

    st.write(
        'Un score **supérieur ou égal à 91 / 100** est nécessaire pour accorder un crédit.'
    )
    
    st.write(
        f'Le score du client est de : **{round(pred)}**'
    )

    if pred >= 91:
        st.markdown(
            '## <center> 🟢 Félicitaion le crédit est sans risque 🟢 </center>',
            unsafe_allow_html=True
        )

    st.markdown(
        '### Données du client :'
    )

    if pred < 91:
        st.markdown(
            '## <center> 🔴 Malheuresement le crédit est à risque 🔴 </center>',
            unsafe_allow_html=True
        )
    
    col = st.multiselect(
        label = 'Caractéristiques du clients à choisir',
        options = data_cli.columns
    )

    data_cli_T = data_cli.T
    data_cli_T['Moyenne non risque'] = mean_zero
    data_cli_T['Moyenne risque'] = mean_un
    data_cli_T['Moyenne ensemble clients'] = mean_data

    if col == []:      
        st.dataframe(
            data_cli_T
        )
    if col != []:
        st.dataframe(
            data_cli_T.loc[col]
        )

    st.write(
        '## Explication du modèle'
    )

    nb_var = st.slider(
        'Nombre de variables',
        1,
        133,
        10
    )

    st.pyplot(
        shap.plots.waterfall(shap_values[ind], max_display=nb_var)
    )
           

        # st.write(f'Score : {pred:.2f}')
        # st.write(f'{pred}')
        
if __name__ == "__main__":
    main()