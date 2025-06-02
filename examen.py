import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import os
import gdown

st.title('Análisis Estadístico de Crímenes en la ciudad de "Los Angeles"')

@st.cache_data
def load_data():
    file_id = '1-skQNlFb1w4RIRQRG4-WH-txfQA3RtlP'  
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'Crime_Data_from_2020_to_Present.csv'

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    
    df = pd.read_csv(output, nrows=90000)

    column_translation = {
        'DR_NO': 'Número Reporte',
        'Date Rptd': 'Fecha Reporte',
        'DATE OCC': 'Fecha Ocurrencia',
        'TIME OCC': 'Hora Ocurrencia',
        'AREA': 'Código Área',
        'AREA NAME': 'Área',
        'Rpt Dist No': 'Distrito Reporte',
        'Part 1-2': 'Clasificación Crimen',
        'Crm Cd': 'Código Crimen',
        'Crm Cd Desc': 'Descripción Crimen',
        'Mocodes': 'Modus Operandi',
        'Vict Age': 'Edad Víctima',
        'Vict Sex': 'Sexo Víctima',
        'Vict Descent': 'Descendencia Víctima',
        'Premis Cd': 'Código Lugar',
        'Premis Desc': 'Descripción Lugar',
        'Weapon Used Cd': 'Código Arma',
        'Weapon Desc': 'Descripción Arma',
        'Status': 'Código Estado',
        'Status Desc': 'Descripción Estado',
        'Crm Cd 1': 'Código Crimen 1',
        'Crm Cd 2': 'Código Crimen 2',
        'Crm Cd 3': 'Código Crimen 3',
        'Crm Cd 4': 'Código Crimen 4',
        'LOCATION': 'Dirección',
        'Cross Street': 'Calle Cercana',
        'LAT': 'Latitud',
        'LON': 'Longitud'
    }
    df.rename(columns=column_translation, inplace=True)
    df = df.dropna(subset=['Edad Víctima', 'Sexo Víctima', 'Latitud', 'Longitud'])
    df = df[df['Sexo Víctima'].isin(['M', 'F'])]
    df['Edad Víctima'] = pd.to_numeric(df['Edad Víctima'], errors='coerce')

    df_completo = df.dropna(subset=['Código Arma', 'Descripción Arma', 'Código Estado', 'Descripción Estado'])
    df_valid_age = df[df['Edad Víctima'] > 0]

    return df, df_completo, df_valid_age

df, df_completo, df_valid_age = load_data()
df_sample = df.sample(n=5000, random_state=42)  # para mapas, usa solo 5000 para rendimiento porque streamlit se cuelga

st.header(' Dataset (primeras 100 filas)')
st.dataframe(df.head(100))

# Top 10 tipos de crimen y áreas
st.write('**Top 10 tipos de crimen:**', df['Descripción Crimen'].value_counts().head(10))
st.write('**Top 10 áreas con más crímenes:**', df['Área'].value_counts().head(10))

# Gráficos
st.subheader('Crímenes por Área')
fig_area, ax = plt.subplots(figsize=(8, 4))
df['Área'].value_counts().head(10).plot(kind='barh', color='skyblue', ax=ax)
ax.set_title('Top 10 Áreas con más Crímenes')
st.pyplot(fig_area)

st.subheader('Distribución de Edad de las Víctimas')
fig_age, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_valid_age['Edad Víctima'], bins=30, color='purple', ax=ax)
ax.set_title('Distribución de Edad de las Víctimas')
st.pyplot(fig_age)

# Mapas por sexo 
st.header('Mapas de Crímenes Agrupados por Sexo')

with st.container():
    st.subheader('Mapa de Crímenes contra Hombres')
    m_male = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles='OpenStreetMap')
    male_grouped = df_sample[df_sample['Sexo Víctima'] == 'M'].groupby('Área').agg({
        'Latitud': 'mean',
        'Longitud': 'mean',
        'Número Reporte': 'count'
    }).reset_index()
    for _, row in male_grouped.iterrows():
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5 + (row['Número Reporte'] / 50),
            color='blue',
            fill=True,
            fill_opacity=0.5,
            popup=f"Área: {row['Área']}<br>Crímenes: {row['Número Reporte']}"
        ).add_to(m_male)
    st_folium(m_male, width=1200, height=500)

with st.container():
    st.subheader('Mapa de Crímenes contra Mujeres')
    m_female = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles='OpenStreetMap')
    female_grouped = df_sample[df_sample['Sexo Víctima'] == 'F'].groupby('Área').agg({
        'Latitud': 'mean',
        'Longitud': 'mean',
        'Número Reporte': 'count'
    }).reset_index()
    for _, row in female_grouped.iterrows():
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5 + (row['Número Reporte'] / 50),
            color='red',
            fill=True,
            fill_opacity=0.5,
            popup=f"Área: {row['Área']}<br>Crímenes: {row['Número Reporte']}"
        ).add_to(m_female)
    st_folium(m_female, width=1200, height=500)

# Áreas con más crímenes a mujeres y hombres
st.header('Áreas con más crímenes a mujeres y hombres')
top_female_area = df[df['Sexo Víctima']=='F']['Área'].value_counts().idxmax()
top_male_area = df[df['Sexo Víctima']=='M']['Área'].value_counts().idxmax()
st.write(f"**Área con más crímenes a mujeres:** {top_female_area}")
st.write(f"**Área con más crímenes a hombres:** {top_male_area}")

#  Pruebas estadísticas
st.header('Pruebas Estadísticas')
contingencia = pd.crosstab(df['Descripción Crimen'], df['Sexo Víctima'])
chi2, p_chi2, _, _ = chi2_contingency(contingencia)
st.write(f" **Prueba de Chi-cuadrado**: Chi2 = {chi2:.2f}, p-valor = {p_chi2:.6e}")

top_crimes = df_valid_age['Descripción Crimen'].value_counts().head(3).index
samples = [df_valid_age[df_valid_age['Descripción Crimen'] == crime]['Edad Víctima'] for crime in top_crimes]
stat, p_kruskal = kruskal(*samples)
st.write(f" **Prueba de Kruskal-Wallis**: H = {stat:.2f}, p-valor = {p_kruskal:.6e}")

male_age = df_valid_age[df_valid_age['Sexo Víctima'] == 'M']['Edad Víctima']
female_age = df_valid_age[df_valid_age['Sexo Víctima'] == 'F']['Edad Víctima']
t_stat, p_ttest = ttest_ind(male_age, female_age, equal_var=False)
st.write(f" **Prueba T-Test**: t = {t_stat:.2f}, p-valor = {p_ttest:.6e}")

# Conclusiones
st.header('Conclusiones de las Pruebas Estadísticas')
st.markdown("""
 **Conclusión de la Prueba de Chi-cuadrado:**  
Encontramos que sí existe una relación importante entre el tipo de crimen y el sexo de la víctima. Esto significa que algunos tipos de crímenes afectan más a hombres o más a mujeres.

 **Conclusión de la Prueba de Kruskal-Wallis:**  
Vemos que hay diferencias significativas en la edad de las víctimas según el tipo de crimen. Por ejemplo, ciertos crímenes son más frecuentes en jóvenes, mientras que otros ocurren más en personas de mayor edad.

 **Conclusión de la Prueba T-Test:**  
Observamos que la edad promedio de las víctimas es diferente entre hombres y mujeres. Es decir, las víctimas hombres y mujeres suelen tener edades distintas en los registros de crímenes.

 **Interpretación final:**  
Estos resultados nos muestran que los crímenes no afectan a todos por igual: el sexo y la edad de las víctimas varían según el tipo de crimen. Esta información puede ayudar a diseñar medidas de seguridad o prevención más específicas para cada grupo.
""")
