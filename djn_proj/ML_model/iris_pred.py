import joblib
import numpy as np

# 1. 저장된 모델 불러오기
model = joblib.load('svc_model.pkl')

# 2. 임의의 입력값 (예: 4개의 특성을 가진 데이터)
# Iris 데이터셋에서 각 특성은 [sepal length, sepal width, petal length, petal width]
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # 예시 값 (5.1, 3.5, 1.4, 0.2)

# 3. 예측 수행
prediction = model.predict(new_data)

# 4. 예측 결과 출력
print(f"Predicted class: {prediction[0]}")

# 5. 예측된 클래스에 해당하는 이름 출력 (Iris dataset의 클래스 이름은 0: Setosa, 1: Versicolor, 2: Virginica)
iris_class_names = ['Setosa', 'Versicolor', 'Virginica']
print(f"Predicted species: {iris_class_names[prediction[0]]}")
