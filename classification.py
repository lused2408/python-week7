import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Đọc dữ liệu (CHÚ Ý sep=';')
df = pd.read_csv(r'c:\Users\Admin\pthon\week7\student-mat.csv', sep=';')

print("Đọc file OK")
print(df.head())

# 2. Tạo cột Pass/Fail
df['Pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# 3. Tách dữ liệu
X = df.drop(['G3', 'Pass'], axis=1)
y = df['Pass']

# 4. Encode
X = pd.get_dummies(X)

print("Shape:", X.shape)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Train xong")

# 7. Predict
y_pred = model.predict(X_test)

# 8. Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)