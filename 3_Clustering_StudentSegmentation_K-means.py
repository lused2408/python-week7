import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Đang tiến hành phân cụm học sinh...")

# 1. Tải dữ liệu
try:
    df = pd.read_csv('student-mat.csv', sep=';')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file student-mat.csv. Phu nhân kiểm tra lại đường dẫn nhé!")
    exit()

# 2. Tiền xử lý dữ liệu
# K-means chỉ nhận số, nên ta tiếp tục dùng get_dummies để chuyển chữ thành số
df_numeric = pd.get_dummies(df, drop_first=True)

# Chuẩn hóa dữ liệu: Đưa tất cả các cột về cùng một thang đo 
# (Rất quan trọng cho K-means để tính khoảng cách chính xác)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# 3. Áp dụng thuật toán K-means
# Đệ tạm chia học sinh thành 3 nhóm (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 4. Gán nhãn nhóm vừa phân loại vào tập dữ liệu ban đầu
df['Cluster'] = kmeans.labels_

# 5. Phân tích kết quả
print("-" * 40)
print("Số lượng học sinh trong từng nhóm:")
print(df['Cluster'].value_counts().sort_index())
print("-" * 40)

# Xem điểm trung bình (G1, G2, G3) và số ngày vắng mặt (absences) của từng nhóm
print("Đặc điểm trung bình của từng nhóm:")
summary = df.groupby('Cluster')[['G1', 'G2', 'G3', 'absences']].mean().round(2)
print(summary)