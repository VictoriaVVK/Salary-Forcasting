import numpy as np
import pandas as pd

def generate_salary_dataset(n=300, seed=42):
    np.random.seed(seed)

    # Реалистични диапазони за факторите
    YearsExperience = np.random.randint(0, 21, n)     # 0–20 години опит
    EducationLevel = np.random.randint(1, 5, n)       # 1=Средно, 4=Доктор
    JobLevel = np.random.randint(1, 5, n)             # 1=Junior, 4=Lead

    # Реалистична формула за заплата, базирана на публични HR модели
    Salary = (
        1800 +                          # базова заплата
        YearsExperience * 320 +         # влияние на опита
        EducationLevel * 1300 +         # влияние на образованието
        JobLevel * 2200 +               # влияние на длъжността
        np.random.normal(0, 1800, n)    # шум (вариация)
    )

    df = pd.DataFrame({
        "YearsExperience": YearsExperience,
        "EducationLevel": EducationLevel,
        "JobLevel": JobLevel,
        "Salary": Salary
    })

    df.to_csv("salary_data.csv", index=False)
    print("🎉 Генериран е реалистичен Salary Dataset с", len(df), "записа!")
    print(df.head())

if __name__ == "__main__":
    generate_salary_dataset()
