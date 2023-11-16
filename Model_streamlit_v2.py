import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np
import altair as alt

st.title('Uso del modelo')

# Cargar los modelos desde los archivos
xgboost_CPC = xgb.XGBRegressor()
xgboost_CPC.load_model("model_xgboost_CPC_v2.json")

xgboost_CPM = xgb.XGBRegressor()
xgboost_CPM.load_model("model_xgboost_CPM_v2.json")

xgboost_CTR = xgb.XGBRegressor()
xgboost_CTR.load_model("model_xgboost_CTR_v2.json")

rf_CPV = joblib.load("model_rf_CPV_v2.joblib")


# Cargar el scaler desde el archivo
loaded_scaler = joblib.load('scaler_model_v2.joblib')
# Cargar el pca desde el archivo
loaded_pca = joblib.load('pca_model_v2.joblib')

# Cargar el scaler desde el archivo
loaded_scaler_CPV = joblib.load('scaler_model_CPV_v2.joblib')
# Cargar el pca desde el archivo
loaded_pca_CPV = joblib.load('pca_model_CPV_v2.joblib')

def load_data(df_in):
    df = pd.read_csv(df_in+'.csv')
    df = df.drop("Unnamed: 0", axis=1)
    return df

# Cargar los datos
df = load_data('df_histo')

def load_clients(df_in):
    df = pd.read_csv(df_in+'.csv')
    return df

df_clients = load_clients('Clients')
df_search_bench = load_clients('search_bench')
df_search_bench = df_search_bench.set_index('Tipo Search')

df_FB_bench = load_clients('FB_bench')
df_FB_bench = df_FB_bench.set_index('Tipo FB')

df_YT_bench = load_clients('YT_bench')
df_YT_bench = df_YT_bench.set_index('Tipo YT')




variables_modelo = xgboost_CPC.feature_names_in_

all_features = ['Año','Mes', 'Objective', 'Cost', 'Country', 'Media_type', 'Traffic_source','Format_New','Platform','Strategy','Plataforma'
                ,'Campaign_Type','Ecommerce','Service_Product','Semanas_Antiguedad','Client','Bench Gral CPC','Bench Search CPC','Bench GralSch CPL', 'Bench Search CPL',
                'Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR','Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction',
                'Bench GralFB CTR', 'Bench FB CTR', 'Bench GralFB CR', 'Bench FB AvgCR','Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']

categorical_features = ['Objective', 'Country', 'Media_type', 'Traffic_source', 'Format_New','Platform','Strategy','Plataforma','Campaign_Type','Ecommerce','Service_Product','Client']



with st.sidebar:
    
    
    Año = st.number_input('Año de la campaña a planificar',value=2023)
    Mes = st.number_input('Mes de la campaña a planificar',value=11, max_value=12)
    Cost = st.number_input('Costo mensual de la campaña',value=1000)
    Objective = st.selectbox(    'Objetivo',    (['Purchase','Fans','Reach', 'Traffic', 'Category', 'Awareness','Product', 'Consideration',
                                                  'Conversion', 'Views','Landing Page Views', 'NoObjective', 'Discovery', 'Impressions','Clicks', 'Conversions', 'Whatsapp']))
    Country = st.selectbox(    'Country',    (['USA','Mexico', 'Chile', 'Colombia', 'Perú', 'Ecuador', 'Argentina']))
    Media_type = st.selectbox(    'Media_type',    (['Search','Social', 'Unknown', 'Display']))
    Traffic_source = st.selectbox(    'Traffic_source',    (['Google','Facebook',  'Other', 'LinkedIn']))
    dict_client = {'HN': 'Hughesnet', 'BR': 'Braun', 'EP': 'Enterprise', 'QQ':'QuickQuack', 'CJ':'ChefJames','OG':'OldGlory', 'AV':'AOV','Nuevo':'Nuevo'}
    
    Tipo_Search = st.selectbox(    'Industria_A',    (["Home - Home Improvement", "Animals - Pets",    "Apparel - Fashion",    "Arts - Entertainment",    "Attorneys - Legal Services",    "Automotive - For sale",
                                                       "Automotive - Repair, Service and Parts",    "Beauty - Personal Care",    "Business Services",    "Career - Employment",
                                                       "Dentists - Dental Services",    "Education - Instruction",    "Finance - Insurance",    "Furniture",    "Health - Fitness",
                                                       "Industrial - Commercial",    "Personal Services",    "Physicians - Surgeons",    "Real Estate",
                                                       "Restaurants - Food",    "Shopping, Collectibles and gifts",    "Sports and Recreation", "Travel",    "Ninguna de las anteriores"     ]))
    Tipo_FB = st.selectbox(    'Industria_B',([  'Consumer Services',   'Apparel',    'Automotive',    'Beauty',       'Education',    'Employment & Job Training',    'Finance & Insurance',
                                                       "Fitness",    "Home Improvement",    "Healthcare",    "Industrial Services",    "Legal",    "Real Estate",    "Retail",    "Technology",
                                                       "Travel & Hospitality",    "Ninguna de las anteriores" ]))





    Tipo_YT = st.selectbox(    'Industria_C',    ([ "Telecommunications",    "Education",    "Fashion",       "Healthcare & Insurance",    "Household Appliances",
                                                           "Entertainment",    "Restaurants",    "Government & Advocacy",    "Health & Beauty",    "Travel",    "Business & Finance",
                                                           "Science & Technology",    "Automotive",    "Electronics",    "Home & Garden",    "Retail",    "Ninguna de las anteriores"]))
                           
    Client = st.selectbox(    'Client',    (['HN', 'BR', 'EP', 'QQ', 'CJ','OG', 'AV','Nuevo'])) #['Hughesnet', 'Braun', 'Enterprise', 'QuickQuack', 'ChefJames','OldGlory', 'AOV']
    Client = dict_client[Client]
    Format_New = st.selectbox(    'Format_New',    (['Display', 'Video']))
    Platform = st.selectbox(    'Platform',    (['Google Ads','Search','Facebook&Instagram', 'Discovery', 'Facebook', 'Performance Max','NoPlatform',  'Facebook & Instagram', 'Programmatic','Google Ads Search', 'LinkedIn','Google Ads Display', 'Google Ads  PMAX']))
    Strategy = st.selectbox(    'Strategy',    (['Consideration','Awareness', 'Conversion',  'Views', 'NoStrategy']))
    Plataforma = st.selectbox(    'Plataforma',    (['Google Ads','Meta',  'External Source', 'NoPlataforma']))
    Campaign_Type = st.selectbox(    'Campaign_Type',    (['SEARCH','PAGE_LIKES', 'DISCOVERY', 'OUTCOME_LEADS', 'CONVERSIONS','LINK_CLICKS', 'PERFORMANCE_MAX',  'OUTCOME_AWARENESS',
                                                           'REACH', 'OUTCOME_SALES', 'NoType', 'DISPLAY','OUTCOME_ENGAGEMENT']))
    Ecommerce = st.selectbox(    'Ecommerce',    (['Si','No']))
    Service_Product = st.selectbox(    'Service_Product',    (['Serv','Prod']))

    new_data = pd.DataFrame({
    'Año': [Año],
    'Mes': [Mes],
    'Objective': [Objective],
    'Cost': [Cost/4],
    'Country': [Country],
    'Media_type': [Media_type],
    'Traffic_source': [Traffic_source],
    'Client': [Client],
    'Format_New': [Format_New],
    'Platform': [Platform],
    'Strategy': [Strategy],
    'Plataforma': [Plataforma],
    'Campaign_Type': [Campaign_Type],
    'Ecommerce': [Ecommerce],
    'Service_Product': [Service_Product],
    'Bench Gral CPC': df_search_bench.loc["Ninguna de las anteriores", "Bench Search CPC"],
    'Bench Search CPC': df_search_bench.loc[Tipo_Search, "Bench Search CPC"],
    'Bench GralSch CPL': df_search_bench.loc["Ninguna de las anteriores", "Bench Search CPL"],
    'Bench Search CPL': df_search_bench.loc[Tipo_Search, "Bench Search CPL"],
    'Bench GralSch CTR': df_search_bench.loc["Ninguna de las anteriores", "Bench Search CTR"],
    'Bench Search CTR': df_search_bench.loc[Tipo_Search, "Bench Search CTR"],
    'Bench GralSch CR': df_search_bench.loc["Ninguna de las anteriores", "Bench Search AvgCR"],
    'Bench Search AvgCR': df_search_bench.loc[Tipo_Search, "Bench Search AvgCR"],
    'Bench GralFB CPC': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB CPC"],
    'Bench FB CPC': df_FB_bench.loc[Tipo_FB, "Bench FB CPC"],
    'Bench GralFB CPAction': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB CPAction"],
    'Bench FB CPAction': df_FB_bench.loc[Tipo_FB, "Bench FB CPAction"],
    'Bench GralFB CTR': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB CTR"],
    'Bench FB CTR': df_FB_bench.loc[Tipo_FB, "Bench FB CTR"],
    'Bench GralFB CR': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB AvgCR"],
    'Bench FB AvgCR': df_FB_bench.loc[Tipo_FB, "Bench FB AvgCR"],
    'Bench GralYT CPV': df_YT_bench.loc["Ninguna de las anteriores","Bench YT CPV"],
    'Bench YT CPV': df_YT_bench.loc[Tipo_YT, "Bench YT CPV"],
    'Bench GralYT CTR': df_YT_bench.loc["Ninguna de las anteriores", "Bench YT CTR"],
    'Bench YT CTR': df_YT_bench.loc[Tipo_YT, "Bench YT CTR"],
    'Bench GralYT VR': df_YT_bench.loc["Ninguna de las anteriores", "Bench FB AvgVR"],
    'Bench FB AvgVR': df_YT_bench.loc[Tipo_YT, "Bench FB AvgVR"]
    })
    
    
    # Preprocesamiento de variables categóricas
    X = pd.get_dummies(new_data, columns=categorical_features)
    
    # Asegurarte de que 'new_data_encoded' tenga las mismas columnas que se utilizaron durante el entrenamiento
    for col in variables_modelo:
        if col not in X.columns:
            X[col] = False  # Agregar la columna faltante con valores predeterminados si es necesario

    
    X_Scaled = loaded_scaler.transform(X[['Año','Mes','Cost','Bench Gral CPC','Bench Search CPC', 'Bench GralSch CPL', 'Bench Search CPL','Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR',
                                          'Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction', 'Bench GralFB CTR','Bench FB CTR',
                                          'Bench GralFB CR', 'Bench FB AvgCR',       'Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']])
    X_pca = loaded_pca.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X['X_pca_0'] = X_pca[0]
    X['X_pca_1'] = X_pca[1]

    # Me aseguro de que mi nuevo dato tiene las mismas variables y el mismo orden que la data con la que fue entrenado el modelo
    X = X[variables_modelo]
    X.columns = X.columns.astype(str)

def prediccion_modelo(modelo,X):
    return modelo.predict(X)

bin_density = st.slider('Bins', min_value=250, max_value=350, step=5, value=300)

st.button("Reset", type="primary")
if st.button('Hacer predicción'):
    pred_CPC = prediccion_modelo(xgboost_CPC,X)[0]
    pred_CPM = prediccion_modelo(xgboost_CPM,X)[0] 
    pred_CTR = prediccion_modelo(xgboost_CTR,X)[0]

    #CPV
    for col in rf_CPV.feature_names_in_:
        if col not in X.columns:
            X[col] = False  # Agregar la columna faltante con valores predeterminados si es necesario
            print(col)

    
    
    X_Scaled = loaded_scaler_CPV.transform(X[['Año','Mes','Cost','Bench Gral CPC','Bench Search CPC', 'Bench GralSch CPL', 'Bench Search CPL','Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR',
                                          'Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction', 'Bench GralFB CTR','Bench FB CTR',
                                          'Bench GralFB CR', 'Bench FB AvgCR',       'Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']])
    X_pca = loaded_pca_CPV.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X['X_pca_0'] = X_pca[0]
    X['X_pca_1'] = X_pca[1]
    
    X = X[rf_CPV.feature_names_in_]
    X.columns = [str(i) for i in X.columns]

    
    
    pred_CPV = prediccion_modelo(rf_CPV,X)[0]

    def histo(df,metrica,valor,bins=bin_density):
        chart = alt.Chart(df).mark_bar(
        opacity=0.3,
        binSpacing=0
    ).encode(
        alt.X(metrica+':Q').bin(maxbins=bin_density),
        alt.Y('count()').stack(None),            
    ).properties(
            width=1000,
            height=600
        ).interactive()

        linea_valor = alt.Chart(pd.DataFrame({'valor_linea': [valor]})).mark_rule(color='red').encode(
    x='valor_linea:Q',
    size=alt.value(2)  # Grosor de la línea
)
        return chart + linea_valor
    
    st.write('CPC')
    st.write(round(pred_CPC,3))
    st.altair_chart(histo(df,'CPC',pred_CPC), use_container_width=False, theme=None)
    
    st.write('CPM')
    st.write(round(pred_CPM,3))
    st.altair_chart(histo(df,'CPM',pred_CPM), use_container_width=False, theme=None)
    
    st.write('CTR')
    st.write(round(pred_CTR,3))
    st.altair_chart(histo(df,'CTR',pred_CTR), use_container_width=False, theme=None)

    if Format_New == 'Video':
        st.write('CPV')
        st.write(round(pred_CPV,3))
        st.altair_chart(histo(df,'CPV',pred_CPV,bins=bin_density*5), use_container_width=False, theme=None)
        
else:
    st.write('Prepara tu predicción')
    
