from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. Iris 데이터셋 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. 데이터셋을 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. SVM 모델 초기화 및 학습
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# 4. 테스트 데이터로 예측 수행
y_pred = model.predict(X_test)

# 5. 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 6. 모델 파라미터 저장
joblib.dump(model, 'svc_model.pkl')
print("Model parameters saved to 'svc_model.pkl'")
