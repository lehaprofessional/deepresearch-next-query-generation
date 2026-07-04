import os
import pandas as pd
from pandas.errors import EmptyDataError

def read_tsv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, sep="\t")
    except EmptyDataError:
        print(f"[ПУСТОЙ ФАЙЛ] {path}")
        return None
    except Exception as e:
        print(f"[ОШИБКА ЧТЕНИЯ] {path}: {e}")
        return None

files = [
    "summerschool_task1.csv",
    "summerschool_task2.csv",
    "train_summerschool_task3.csv",
    "train_summerschool_task4.csv",
]

print("Файлы в папке:")
for name in os.listdir("."):
    print(" -", name)

print("\nПроверка датасетов:\n")

for file in files:
    if not os.path.exists(file):
        print(f"[НЕТ] {file}")
        continue

    size = os.path.getsize(file)
    print(f"[OK] {file}")
    print("Размер файла в байтах:", size)

    df = read_tsv(file)

    if df is None:
        print("-" * 80)
        continue

    print("Размер таблицы:", df.shape)
    print("Колонки:", list(df.columns))
    print("Первые строки:")
    print(df.head(3))
    print("-" * 80)