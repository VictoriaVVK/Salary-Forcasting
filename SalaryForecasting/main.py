# Прогнозиране на заплати с различни модели върху публичен dataset (UCI Adult via OpenML)
# Автор: Victoria Kostadinova / Основи на ИИ, ТУ-Варна

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

print("⏳ Зареждам публичния dataset от OpenML...")
data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame
print("✅ Dataset зареден успешно! Размер:", df.shape)

# --- 1. Почистване ---
df = df.dropna()

# Категориални колони → в числа (Label Encoding)
categorical_cols = ["workclass", "education", "marital-status",
                    "occupation", "relationship", "race", "sex", "native-country"]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# --- 2. Създаване на реалистична заплата ---
# Формула: реалистичен многофакторен модел
np.random.seed(42)

df["Salary"] = (
    15000                                        # базова заплата
    + df["age"] * 350                            # възраст → умерен ефект
    + df["education-num"] * 1800                 # образование → силен ефект
    + df["hours-per-week"] * 25                  # работни часове → директен ефект
    + np.random.normal(0, 8000, len(df))         # шум
)

print("\nПримерни данни:")
print(df[["age", "education-num", "hours-per-week", "Salary"]].head())

# --- 3. Избор на входни характеристики ---
X = df[["age", "education-num", "hours-per-week"]]
y = df["Salary"]

# --- 4. Разделяне ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Модели ---
models = {
    "Линейна регресия": LinearRegression(),
    "Дърво на решенията": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = {}

print("\n📊 Обучаване на моделите...\n")

# --- 6. Обучение и оценка ---
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"model": model, "MSE": mse, "R2": r2}

    print(f"{name}:")
    print(f"  - MSE = {mse:.2f}")
    print(f"  - R²  = {r2:.3f}\n")

# --- 7. Най-добър модел ---
best_model_name = max(results, key=lambda m: results[m]["R2"])
best_model = results[best_model_name]["model"]

print(f"🏆 Най-добър модел според R²: {best_model_name}")

# --- 8. Вход от потребителя ---
print("\nВъведи свои характеристики за прогнозиране:")

age = float(input("Възраст: "))

print("\nИзбери образование:")
print("1 = High School")
print("2 = Associate Degree")
print("3 = Bachelor")
print("4 = Master")
print("5 = Doctorate")

edu_choice = int(input("Въведи номер на образование (1–5): "))

# Превръщане в реалните education-num стойности от Adult dataset
education_map = {
    1: 9,   # High school
    2: 10,  # Associate
    3: 13,  # Bachelor
    4: 14,  # Master
    5: 16   # Doctorate
}

edu = education_map.get(edu_choice, 9)  # default HS

hours = float(input("Часове работа седмично: "))

user_df = pd.DataFrame([{
    "age": age,
    "education-num": edu,
    "hours-per-week": hours
}])

prediction = best_model.predict(user_df)[0]

print(f"\n💰 Прогнозирана годишна заплата: {prediction:.2f} USD\n")

# --- 9. Визуализация: зависимост възраст → заплата ---
plt.style.use("seaborn-v0_8-whitegrid")

lin_model = results["Линейна регресия"]["model"]

plt.figure(figsize=(10, 6))
plt.scatter(df["age"], df["Salary"], alpha=0.3, label="Реални данни", color="blue")

ages = np.linspace(df["age"].min(), df["age"].max(), 200)
edu_mean = df["education-num"].mean()
hours_mean = df["hours-per-week"].mean()

plt.plot(
    ages,
    lin_model.predict(pd.DataFrame({
        "age": ages,
        "education-num": [edu_mean]*200,
        "hours-per-week": [hours_mean]*200
    })),
    color="red",
    label="Линейна регресия"
)

plt.title("Линейна регресия: Възраст → Заплата")
plt.xlabel("Възраст")
plt.ylabel("Годишна заплата (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# --- 10. Бар диаграма: сравнение ---
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), [results[k]["R2"] for k in results], color=["red", "green", "purple"])
plt.title("Сравнение на моделите по точност (R²)")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
