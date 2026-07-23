import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AKDS - Afet Karar Destek Sistemi", layout="wide")

@st.cache_data
def load_data():
    # Gerçek ilçe koordinatları (Haritadaki noktaların doğru görünmesi için sabitlendi)
    city_info = {
        "İstanbul": {
            "coords": {
                "Avcılar": (40.9799, 28.7217),
                "Küçükçekmece": (41.0000, 28.7700),
                "Bakırköy": (40.9833, 28.8667),
                "Zeytinburnu": (40.9900, 28.9000),
                "Fatih": (41.0186, 28.9400),
                "Beşiktaş": (41.0430, 29.0070),
                "Üsküdar": (41.0260, 29.0150),
                "Kadıköy": (40.9900, 29.0250),
                "Pendik": (40.8750, 29.2330),
                "Ümraniye": (41.0250, 29.0960),
                "Esenyurt": (41.0340, 28.6800),
                "Silivri": (41.0730, 28.2470)
            }
        },
        "Bursa": {
            "coords": {
                "Osmangazi": (40.1828, 29.0667),
                "Yıldırım": (40.1900, 29.0900),
                "Nilüfer": (40.2167, 28.9833),
                "Gemlik": (40.4300, 29.1500),
                "Mudanya": (40.3760, 28.8820),
                "İnegöl": (40.0780, 29.5130),
                "Kestel": (40.1980, 29.2120),
                "Gürsu": (40.2180, 29.1930),
                "Karacabey": (40.2150, 28.3580),
                "Mustafakemalpaşa": (40.0350, 28.4110)
            }
        },
        "Kocaeli": {
            "coords": {
                "İzmit": (40.7656, 29.9405),
                "Gebze": (40.8028, 29.4306),
                "Gölcük": (40.7180, 29.8240),
                "Körfez": (40.7760, 29.7370),
                "Derince": (40.7560, 29.8310),
                "Başiskele": (40.7100, 29.9300)
            }
        },
        "Yalova": {
            "coords": {
                "Merkez": (40.6558, 29.2667),
                "Çiftlikköy": (40.6630, 29.3000),
                "Altınova": (40.6930, 29.5090),
                "Çınarcık": (40.6420, 29.1170)
            }
        },
        "Sakarya": {
            "coords": {
                "Adapazarı": (40.7755, 30.4020),
                "Serdivan": (40.7630, 30.3660),
                "Erenler": (40.7580, 30.4330),
                "Akyazı": (40.6850, 30.6230),
                "Hendek": (40.7980, 30.7480)
            }
        }
    }

    all_cities_data = []
    np.random.seed(42)

    for city, info in city_info.items():
        districts = list(info["coords"].keys())
        num_neighborhoods = len(districts)

        latitudes = [info["coords"][d][0] for d in districts]
        longitudes = [info["coords"][d][1] for d in districts]

        city_data = {
            "Sehir": [city] * num_neighborhoods,
            "Mahalle": districts,
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
st.markdown("<h3 style='text-align: center;'>İstanbul'da 7.5 büyüklüğünde deprem meydana geldi.</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Yardım gönderilecek şehri seçiniz.</h4>", unsafe_allow_html=True)
st.divider()

selected_city = st.selectbox("Yardım Gönderilecek Şehri Seçiniz:", df['Sehir'].unique())

deprem_buyuklukleri = {
    "İstanbul": 7.5,
    "Bursa": 6.8,
    "Kocaeli": 7.0,
    "Yalova": 6.6,
    "Sakarya": 6.5
}
magnitude = deprem_buyuklukleri.get(selected_city, 7.0)

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

st.subheader(f"🚨 {selected_city} şehrinde {magnitude} büyüklüğünde deprem analizi yapıldı.")
st.error(f"🚨 **EN YÜKSEK ACİL YARDIM ÖNCELİĞİ:** **{df_sehir.iloc[0]['Mahalle'].upper()}**")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Ortalama Mutlak Hata (MAE)", value=f"{mae:.2f}")
with col2:
    st.metric(label="R-Kare (R²) Skoru", value=f"{r2:.2f}")

st.dataframe(df_sehir[['Mahalle','Oncelik','Hasar','Nufus','65_plus','Hastane_Uzaklik','Bina_Yasi_Ortalama','Zemin_Tipi','Skor']], use_container_width=True)

st.subheader("🗺️ Canlı Hasar ve Öncelik Haritası")

m = folium.Map(location=[merkez_lat, merkez_lon], zoom_start=11, tiles='OpenStreetMap')

if selected_city == "İstanbul":
    for r in range(2000, 8000, 800):
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
        size = 15
    else:
        color = 'yellow'
        size = 12

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=f"{row['Mahalle']} - Skor: {row['Skor']:.1f}"
    ).add_to(m)

st_folium(m, width=1200, height=500)
