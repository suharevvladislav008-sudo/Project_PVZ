import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Настройка визуализации
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Генерация данных по сотрудникам (Выдача заказов)
np.random.seed(42)
n_records = 1000

employees = ['Сухарев В.А.', 'Бамбуляк Д.В.', 'Бойко Д.С.', 'Селимов С.М.']
dates = [datetime(2026, 9, 1) + timedelta(days=int(i%15), hours=int(np.random.choice([9, 12, 15, 18, 19, 20]))) for i in range(n_records)]

df_staff = pd.DataFrame({
    'timestamp': dates,
    'employee': np.random.choice(employees, n_records, p=[0.4, 0.2, 0.2, 0.2]), # Сухарев работает больше всех
    'orders_issued': np.random.randint(1, 5, n_records),
    'handling_time_sec': np.random.normal(120, 40, n_records).astype(int), # Среднее время выдачи ~120 сек
    'customer_rating': np.random.choice([5, 4, 3, 2, 1], n_records, p=[0.7, 0.15, 0.08, 0.05, 0.02])
})

df_staff.loc[df_staff['handling_time_sec'] < 10, 'handling_time_sec'] = 5
df_staff.loc[df_staff['handling_time_sec'] > 600, 'handling_time_sec'] = 900

# 2. Генерация данных инвентаризации (План vs Факт)
n_items = 200
df_inventory = pd.DataFrame({
    'item_id': [f'WB-{10000 + i}' for i in range(n_items)],
    'category': np.random.choice(['Одежда', 'Электроника', 'Крупногабаритный товар (КГТ)', 'Косметика'], n_items, p=[0.5, 0.1, 0.15, 0.25]),
    'planned_qty': np.random.randint(1, 10, n_items)
})
# Создаем расхождения (факт отличается от плана)
df_inventory['actual_qty'] = df_inventory['planned_qty']
mismatch_indices = np.random.choice(n_items, size=30, replace=False)
df_inventory.loc[mismatch_indices, 'actual_qty'] += np.random.choice([-1, 1, -2], size=30)

print("Данные успешно сгенерированы! Размеры таблиц:", df_staff.shape, df_inventory.shape)

# --- Очистка аномалий ---
# Удаляем технические сбои (время выдачи меньше 15 сек и больше 10 минут)
cleaned_staff = df_staff[(df_staff['handling_time_sec'] >= 15) & (df_staff['handling_time_sec'] <= 600)].copy()

# --- Расчет KPI сотрудников ---
# Группируем данные для оценки эффективности каждого работника
employee_kpi = cleaned_staff.groupby('employee').agg(
    total_orders=('orders_issued', 'sum'),
    median_time_sec=('handling_time_sec', 'median'),
    avg_rating=('customer_rating', 'mean')
).reset_index()

print("=== KPI СОТРУДНИКОВ ПВЗ ===")
print(employee_kpi.round(2).to_string(index=False))

# --- Анализ пиковых нагрузок по часам ---
cleaned_staff['hour'] = cleaned_staff['timestamp'].dt.hour

plt.figure(figsize=(10, 5))
sns.countplot(data=cleaned_staff, x='hour', palette='viridis')
plt.title('Распределение потока клиентов по часам (Выявление пиковых нагрузок)')
plt.xlabel('Час суток')
plt.ylabel('Количество обращений')
plt.show()

# --- Поиск расхождений (Сверка остатков) ---
# Находим разницу между планом и фактом
df_inventory['mismatch'] = df_inventory['actual_qty'] - df_inventory['planned_qty']
df_failures = df_inventory[df_inventory['mismatch'] != 0].copy()

# Разделяем на излишки и недостачу
df_failures['status'] = np.where(df_failures['mismatch'] > 0, 'Излишек', 'Недостача')
df_failures['abs_mismatch'] = df_failures['mismatch'].abs()

print(f"Всего позиций с ошибками учета: {len(df_failures)} из {len(df_inventory)}")

# --- Анализ проблемных категорий товаров ---
category_issues = df_failures.groupby(['category', 'status']).agg(
    total_items_error=('abs_mismatch', 'sum')
).reset_index()

# Визуализация ошибок по категориям
plt.figure(figsize=(11, 5))
sns.barplot(data=category_issues, x='category', y='total_items_error', hue='status', palette='magma')
plt.title('Анализ ошибок инвентаризации по категориям товаров')
plt.xlabel('Категория товара')
plt.ylabel('Количество ошибочных позиций (шт)')
plt.xticks(rotation=15)
plt.legend(title='Тип ошибки')
plt.tight_layout()
plt.show()