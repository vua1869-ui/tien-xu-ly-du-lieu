import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ITA105_Lab_1.csv')

print("=" * 50)
print("BÀI 1: KHÁM PHÁ DỮ LIỆU")
print("=" * 50)

print(f"Số dòng : {df.shape[0]}")
print(f"Số cột  : {df.shape[1]}")
print(f"Shape   : {df.shape}")

print("\n--- Thống kê mô tả ---")
print(df.describe())

print("\n--- Giá trị thiếu ---")
print(df.isnull().sum())
print(f"Tổng số ô thiếu: {df.isnull().sum().sum()}")
print("\n" + "=" * 50)
print("BÀI 2: XỬ LÝ DỮ LIỆU THIẾU")
print("=" * 50)
df_dropped = df.dropna()
print(f"Sau dropna : {len(df_dropped)} dòng")
df['Price'] = df['Price'].fillna(df['Price'].mean())
df['StockQuantity'] = df['StockQuantity'].fillna(df['StockQuantity'].median())
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

print(f"Sau fillna : {len(df)} dòng")

print("\n--- Giá trị thiếu sau khi xử lý ---")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("BÀI 3: XỬ LÝ DỮ LIỆU LỖI")
print("=" * 50)

print(f"Price <= 0: {(df['Price'] <= 0).sum()} dòng")
df['Price'] = df['Price'].apply(lambda x: abs(x) if x < 0 else x)

print(f"StockQuantity < 0: {(df['StockQuantity'] < 0).sum()} dòng")
df['StockQuantity'] = df['StockQuantity'].apply(lambda x: 0 if x < 0 else x)

print(f"Rating không hợp lệ (ngoài 0-5): {((df['Rating'] < 0) | (df['Rating'] > 5)).sum()} dòng")
df = df[(df['Rating'] >= 0) & (df['Rating'] <= 5)]

print("\n--- Sau khi xử lý lỗi ---")
print(df[['Price', 'StockQuantity', 'Rating']].describe())

print("\n" + "=" * 50)
print("BÀI 4: LÀM MƯỢT DỮ LIỆU NHIỄU")
print("=" * 50)

df['Price_MA'] = df['Price'].rolling(window=5).mean()
print("Đã tính Moving Average (window=5) cho cột Price")
print(df[['Price', 'Price_MA']].head(10))

plt.figure(figsize=(12, 5))
plt.plot(df['ProductID'], df['Price'], label='Price gốc', alpha=0.6, color='blue')
plt.plot(df['ProductID'], df['Price_MA'], label='Moving Average (window=5)', color='red', linewidth=2)
plt.title('Bài 4: Làm mượt dữ liệu Price bằng Moving Average')
plt.xlabel('ProductID')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('bai4_moving_average.png')
plt.show()
print("Đã lưu biểu đồ: bai4_moving_average.png")

print("\n" + "=" * 50)
print("BÀI 5: CHUẨN HÓA DỮ LIỆU")
print("=" * 50)

df['Category'] = df['Category'].str.lower()
print("Category sau khi lower():", df['Category'].unique())

df['Description'] = df['Description'].str.replace(r'[?!.\-]+', '', regex=True).str.strip()
print("\nDescription sau khi làm sạch:")
print(df['Description'].unique())

USD_TO_VND = 25000
df['Price_VND'] = df['Price'] * USD_TO_VND
print(f"\nGiá sau khi đổi sang VND (1 USD = {USD_TO_VND:,} VND):")
print(df[['Price', 'Price_VND']].head())

print("\n✅ HOÀN THÀNH TẤT CẢ 5 BÀI!")