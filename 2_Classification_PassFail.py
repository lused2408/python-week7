import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Đang xử lý dữ liệu cho bài toán Phân loại...")

# 1. Tải dữ liệu (giả định phu nhân đang đứng đúng thư mục chứa file csv)
try:
    df = pd.read_csv('student-mat.csv', sep=';')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file student-mat.csv. Phu nhân nhớ kiểm tra lại đường dẫn nhé!")
    exit()

# 2. Tạo biến mục tiêu mới: Pass/Fail
# Điểm >= 10 là Đậu (1), dưới 10 là Trượt (0)
df['Pass'] = (df['G3'] >= 10).astype(int)

# 3. Tiền xử lý dữ liệu
# Bỏ cột G3 cũ vì ta đã có cột Pass. 
X = df.drop(columns=['G3', 'Pass'])
y = df['Pass']

# Chuyển đổi các cột chữ (như 'GP', 'M', 'F'...) thành số
X = pd.get_dummies(X, drop_first=True)

# 4. Chia tập dữ liệu (80% để học, 20% để kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Huấn luyện mô hình Phân loại (Sử dụng Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
print("-" * 40)
print("Báo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Trượt (0)', 'Đậu (1)']))