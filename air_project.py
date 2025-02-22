import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap
import seaborn as sns
import plotly.express as px
import numpy as np
from datetime import timedelta

# Загрузка данных о проверках воздуха
try:
    air_data = pd.read_csv("air_quality_inspections.csv", sep=";", encoding="utf-8")
    print("Данные о проверках воздуха загружены.")
except FileNotFoundError:
    print("Ошибка: Файл air_quality_inspections.csv не найден!")
    exit()

# Загрузка данных о погоде
try:
    weather_data = pd.read_csv("weather.txt", sep=r"\s+", names=[
        "Станция", "Год_UTC", "Месяц_UTC", "День_UTC", "Час_UTC",
        "Год_Местн", "Месяц_Местн", "День_Местн", "Час_Местн", "Время_Местн",
        "Часовой_Пояз", "Видимость", "Облачность", "Погода", "Направление_Ветра",
        "Скорость_Ветра", "Осадки", "Температура", "Влажность", "Давление"
    ], skiprows=22, engine='python')
    print("Данные о погоде загружены.")
except FileNotFoundError:
    print("Ошибка: Файл weather.txt не найден!")
    exit()

# Преобразование дат в air_data
air_data["Дата"] = pd.to_datetime(air_data["Дата"], format="%d.%m.%Y", errors="coerce")

if air_data["Дата"].isna().sum() > 0:
    print("Предупреждение: Некоторые даты в air_data не были распознаны!")

air_data["Дата_UTC"] = (air_data["Дата"] - timedelta(hours=3)).dt.floor("D")

# Преобразование дат в weather_data
weather_data["Дата_UTC"] = pd.to_datetime(
    weather_data["Год_UTC"].astype(str) + "-" +
    weather_data["Месяц_UTC"].astype(str).str.zfill(2) + "-" +
    weather_data["День_UTC"].astype(str).str.zfill(2) + " " +
    weather_data["Час_UTC"].astype(str).str.zfill(2) + ":00",
    errors="coerce"
)

# Классификация результатов загрязнения
air_data["Уровень_Загрязнения"] = air_data["Результаты"].apply(
    lambda x: 1 if pd.notna(x) and "выявлены повышенные" in str(x).lower() else 0
)

# Объединение данных
combined_data = pd.merge(
    air_data[["Дата_UTC", "Уровень_Загрязнения", "Район", "Долгота", "Широта"]],
    weather_data[["Дата_UTC", "Скорость_Ветра", "Осадки", "Температура"]],
    on="Дата_UTC",
    how="left"
)

combined_data = combined_data.dropna()
print("Данные успешно объединены.")

# Визуализация превышений по месяцам
monthly_exceedances = combined_data.groupby(combined_data["Дата_UTC"].dt.to_period("M"))["Уровень_Загрязнения"].sum()
plt.figure(figsize=(12, 6))
plt.plot(monthly_exceedances.index.to_timestamp(), monthly_exceedances.values, color="blue", label="Превышения")
plt.title("Превышения загрязнения по месяцам")
plt.xlabel("Дата")
plt.ylabel("Количество превышений")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.savefig("exceedances_plot.png")
plt.show()

# Корреляция погодных факторов и загрязнения
plt.figure(figsize=(8, 6))
sns.heatmap(combined_data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Матрица корреляции погодных факторов и загрязнения")
plt.savefig("correlation_matrix.png")
plt.show()

# Подсчёт превышений по районам
district_exceedances = combined_data.groupby("Район")["Уровень_Загрязнения"].sum()

# Визуализация по районам с помощью seaborn
plt.figure(figsize=(15, 7))
sns.barplot(x=district_exceedances.index, y=district_exceedances.values, hue=district_exceedances.index, palette="viridis", legend=False)
plt.title("Превышения загрязнения по районам", fontsize=14)
plt.xlabel("Район", fontsize=12)
plt.ylabel("Количество превышений", fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("district_exceedances.png")
plt.show()

# Группировка по районам и подсчёт среднего уровня загрязнения и погодных факторов
weather_impact = combined_data.groupby("Район").agg({
    "Уровень_Загрязнения": "mean",
    "Скорость_Ветра": "mean",
    "Осадки": "mean",
    "Температура": "mean"
}).reset_index()

# Влияние погодных факторов на загрязнение по районам
plt.figure(figsize=(12, 6))
sns.scatterplot(data=weather_impact, x="Скорость_Ветра", y="Уровень_Загрязнения", size="Осадки", hue="Температура", palette="coolwarm")
plt.title("Влияние погодных факторов на загрязнение по районам")
plt.xlabel("Средняя скорость ветра")
plt.ylabel("Средний уровень загрязнения")
plt.savefig("weather_impact.png")
plt.show()

# Улучшенная карта загрязнений
m = folium.Map(location=[55.75, 37.61], zoom_start=10)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in air_data.iterrows():
    if pd.notna(row["Широта"]) and pd.notna(row["Долгота"]):
        color = "red" if row["Уровень_Загрязнения"] == 1 else "green"
        folium.CircleMarker(
            location=[float(row["Широта"]), float(row["Долгота"])],
            radius=5, color=color, fill=True,
            popup=f"Дата: {row['Дата']}\nРайон: {row['Район']}\nРезультат: {row['Результаты']}"
        ).add_to(marker_cluster)
HeatMap(
    [[float(row["Широта"]), float(row["Долгота"])]
     for idx, row in air_data.iterrows() if pd.notna(row["Широта"]) and pd.notna(row["Долгота"])]
).add_to(m)
m.save("pollution_map.html")

# Дашборд с plotly
fig = px.scatter(combined_data, x="Дата_UTC", y="Уровень_Загрязнения", color="Район", size="Скорость_Ветра",
                 hover_data=["Температура", "Осадки"], title="Интерактивный анализ загрязнения")
fig.write_html("pollution_dashboard.html")

print("Анализ завершен. Результаты сохранены в файлах.")