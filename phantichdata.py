# Nhập các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tải dữ liệu
data = pd.read_csv('./mynewdata.csv')  # Thay 'mynewdata.csv' bằng đường dẫn đến tệp dữ liệu của bạn

# 1. Khám Phá Dữ Liệu
print("Tổng quan về dữ liệu:")
print(data.info())  # Thông tin tổng quát về dữ liệu

# a. Thống kê mô tả
print("\nThống kê mô tả:")
print(data.describe())  # Thống kê mô tả cho các biến số

# b. Phân tích sự thiếu dữ liệu
missing_values = data.isnull().sum()
print("\nGiá trị thiếu trong từng biến:")
print(missing_values[missing_values > 0])  # Chỉ hiển thị các biến có giá trị thiếu

# c. Histogram để phân tích phân phối
plt.figure(figsize=(12, 6))
sns.histplot(data['age'], kde=True)  # Phân phối độ tuổi
plt.title('Phân phối độ tuổi')
plt.xlabel('Tuổi')
plt.ylabel('Tần suất')
plt.show()

# d. Boxplot để phát hiện mẫu ngoại lệ cho từng biến số
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']  # Các biến số cần phân tích

# Lặp qua từng biến số để phát hiện mẫu ngoại lệ và thay đổi giá trị
for feature in numerical_features:
    # Tính toán các ngưỡng cho mẫu ngoại lệ bằng phương pháp IQR
    Q1 = data[feature].quantile(0.25)  # 25th percentile
    Q3 = data[feature].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Khoảng IQR

    # Xác định giới hạn dưới và trên
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Thay đổi giá trị các mẫu ngoại lệ
    data[feature] = data[feature].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

    # In ra các giới hạn để bạn có thể kiểm tra
    print(f"Giới hạn cho {feature}: [{lower_bound}, {upper_bound}]")



print("\nDữ liệu sau khi thay đổi giá trị ngoại lệ và chuẩn hóa:")
print(data.describe())  # Kiểm tra lại thống kê mô tả sau khi thay đổi

# Giả sử DataFrame của bạn được lưu trong biến `df`
data.to_csv("dataxuly.csv", index=False, encoding='utf-8')
