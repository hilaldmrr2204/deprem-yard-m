import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Deprem Acil Durum Sistemi", layout="wide")

@st.cache_data
def load_data():
    city_info = {
        "İstanbul": {
            "districts": ["Avcılar","Küçükçekmece","Bakırköy","Zeytinburnu","Fatih","Beşiktaş","Üsküdar","Kadıköy","Pendik","Ümraniye","Esenyurt","Silivri"],
            "lat_center": 41.0082, "lon_center": 28.9784
        },
        "Bursa": {
            "districts": ["Osmangazi","Yıldırım","Nilüfer","Gemlik","Mudanya","İnegöl","Kestel","Gürsu","Karacabey","Mustafakemalpaşa"],
            "lat_center": 40.1885, "lon_center": 29.0609
        },
        "Kocaeli": {
            "districts": ["İzmit","Gebze","Gölcük","Körfez","Derince","Başiskele"],
            "lat_center": 40.7656, "lon_center": 29.9405
        },
        "Yalova": {
            "districts": ["Merkez","Çiftlikköy","Altınova","Çınarcık"],
            "lat_center": 40.6558, "lon_center": 29.2667
        },
        "Sakarya": {
            "districts": ["Adapazarı","Serdivan","Erenler","Akyazı","Hendek"],
            "lat_center": 40.7755, "lon_center": 30.4020
        }
    }

    all_cities_data = []
    np.random.seed(42)

    for city, info in city_info.items():
        num_neighborhoods = len(info["districts"])
        city_lat_center = info.get("lat_center", 39.0)
        city_lon_center = info.get("lon_center", 35.0)

        latitudes = np.random.uniform(city_lat_center - 0.08, city_lat_center + 0.08, num_neighborhoods)
        longitudes = np.random.uniform(city_lon_center - 0.08, city_lon_center + 0.08, num_neighborhoods)

        city_data = {
            "Sehir": [city] * num_neighborhoods,
            "Mahalle": info["districts"],
            "Hasar": np.random.randint(55, 90, num_neighborhoods),
            "Nufus": np.random.randint(50000, 800000, num_neighborhoods),
            "65_plus": np.random.randint(8, 22, num_neighborhoods),
            "Hastane_Uzaklik": np.round(np.random.uniform(0.5, 8, num_neighborhoods), 1),
            "Yol_Erisim": np.random.randint(0, 2, num_neighborhoods),
            "Bina_Yasi_Ortalama": np.random.randint(8, 45, num_neighborhoods),
            "Zemin_Tipi": np.random.choice(["Saglam", "Orta", "Zayif"], num_neighborhoods, p=[0.35, 0.40, 0.25]),
            "Latitude": latitudes,
            "Longitude": longitudes
        }
        all_cities_data.append(pd.DataFrame(city_data))

    return pd.concat(all_cities_data, ignore_index=True)

df = load_data()

st.markdown("<h1 style='text-align: center; color: red;'>🚨 AKDS - Afet Karar Destek Sistemi 🚨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Yapay Zeka Destekli İlçe Önceliklendirme Haritası</h3>", unsafe_allow_html=True)
st.divider()

st.sidebar.header("Deprem Parametreleri")
selected_city = st.sidebar.selectbox("Şehir Seçiniz:", df['Sehir'].unique())

deprem_buyuklukleri = {
    "İstanbul": 7.5,
    "Bursa": 6.8,
    "Kocaeli": 7.0,
    "Yalova": 6.6,
    "Sakarya": 6.5
}
magnitude = deprem_buyuklukleri.get(selected_city, 7.0)
st.sidebar.info(f"Seçilen Şehir: **{selected_city}**\n\nDeprem Büyüklüğü: **{magnitude}**")

df_sehir = df[df['Sehir'] == selected_city].copy()

merkez_lat = df_sehir['Latitude'].mean()
merkez_lon = df_sehir['Longitude'].mean()

df_sehir['Merkez_Uzaklik'] = np.sqrt(
    (df_sehir['Latitude'] - merkez_lat)**2 +
    (df_sehir['Longitude'] - merkez_lon)**2
)

zemin_mapping = {'Saglam': 1, 'Orta': 2, 'Zayif': 3}
df_sehir['Zemin_Tipi_Kod'] = df_sehir['Zemin_Tipi'].map(zemin_mapping)

df_sehir['Gercek'] = (
    df_sehir['Hasar'] * 0.3 +
    df_sehir['65_plus'] * 0.15 +
    df_sehir['Nufus'] * 0.0003 +
    df_sehir['Hastane_Uzaklik'] * 3 +
    df_sehir['Merkez_Uzaklik'] * 70 +
    df_sehir['Bina_Yasi_Ortalama'] * 0.5 +
    df_sehir['Zemin_Tipi_Kod'] * 10
)

X = df_sehir[['Hasar','Nufus','65_plus','Hastane_Uzaklik','Yol_Erisim','Merkez_Uzaklik', 'Bina_Yasi_Ortalama', 'Zemin_Tipi_Kod']]
y = df_sehir['Gercek']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

df_sehir['Skor'] = model.predict(X)

df_sehir["Oncelik"] = "Orta"
df_sehir.loc[df_sehir["Skor"] >= df_sehir["Skor"].quantile(0.50), "Oncelik"] = "Yüksek"
df_sehir.loc[df_sehir["Skor"] >= df_sehir["Skor"].quantile(0.80), "Oncelik"] = "Kritik"

df_sehir = df_sehir.sort_values(by='Skor', ascending=False)

en_öncelikli = df_sehir.iloc[0]['Mahalle']

st.error(f"🚨 **EN YÜKSEK ACİL YARDIM ÖNCELİĞİ:** **{en_öncelikli.upper()}**")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Yapay Zeka Hata Payı (MAE)", value=f"{mae:.2f}")
with col2:
    st.metric(label="Model Doğruluk Skoru (R²)", value=f"{r2:.2f}")

st.subheader(f"📊 {selected_city} İlçesi/Mahalle Öncelik Analiz Tablosu")
st.dataframe(df_sehir[['Mahalle','Oncelik','Hasar','Nufus','65_plus','Hastane_Uzaklik','Bina_Yasi_Ortalama','Zemin_Tipi','Skor']], use_container_width=True)

st.subheader("🗺️ Canlı Hasar ve Öncelik Haritası")

m = folium.Map(location=[merkez_lat, merkez_lon], zoom_start=11, tiles='OpenStreetMap')

if selected_city == "İstanbul":
    for r in range(2000, 8000, 1500):
        folium.Circle(
            location=[merkez_lat, merkez_lon],
            radius=r,
            color='red',
            fill=True,
            fill_opacity=0.05
        ).add_to(m)

for i, row in df_sehir.iterrows():
    if row['Oncelik'] == 'Kritik':
        color = 'red'
        size = 18
    elif row['Oncelik'] == 'Yüksek':
        color = 'orange'
        size = 14
    else:
        color = 'yellow'
        size = 10

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=f"<b>{row['Mahalle']}</b><br>Öncelik: {row['Oncelik']}<br>Skor: {row['Skor']:.1f}"
    ).add_to(m)

st_folium(m, width=1200, height=500)
