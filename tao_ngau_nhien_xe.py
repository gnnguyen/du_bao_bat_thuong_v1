# từ tập tin data_motobikes.xlsx hãy lấy ra 100 xe máy ngẫu nhiên và đưa vào tập tin subset_100motobikes.csv
import pandas as pd
import random
# Đọc dữ liệu từ file excel
df = pd.read_excel('data_motobikes.xlsx', engine='openpyxl')

# Lấy 100 xe máy ngẫu nhiên
random_bikes = df.sample(n=100, random_state=42, replace=False)
# Lưu vào file CSV
random_bikes.to_csv('subset_100motobikes.csv', index=False)
print("Đã lưu 100 xe máy ngẫu nhiên vào subset_100motobikes.csv")

