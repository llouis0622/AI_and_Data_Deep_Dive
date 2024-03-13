## 1. Image Classification(이미지 분류)
- 사진 -> 숫자의 집합으로 표현
- 실제 컴퓨터가 보는 숫자값 -> 격차 존재 : Semantic Gap
  ```
  def classify_image(image):
    # Some magic here?
    return class_label
  ```
- 이미지 -> edges 추출 -> 변화에 Robust 하지 않음
- Data-Driven Approach(데이터 중심 접근방법) : Machine Learning Classifier 학습

## 2. Data-Driven Approach
- Nearest Neighbor Classifier
  ```
  def train(images, labels):
    # Machine Learning!
    return model
  
  def predict(model, test_images):
    # Use Model to Predict Labels
    return test_labels
  ```
- train 함수 : 이미지와 레이블을 input으로 줌 -> 머신러닝
- predict 함수 : train 함수에서 반환된 모델을 가지고 이미지 판단
- Distance Metric - L1 Distance
  - Manhattan Distance
  - 각 픽셀값 비교, 그 차이의 절댓값을 모두 합하여 하나의 지표로 설정
  ```
  import numpy as np
  
  class NearestNeighbor:
    def __init__(self):
      pass

    def train(self, X, y):
      self.Xtr = X
      self.ytr = y
  
    def predict(self, X):
      num_test = X.shape[0]
      Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
      for i in xrange(num_test):
        distances = np.sum(np.abs(self.Xtr - X[i, :], axis=1)
        min_index = np.argmin(distances)
        Ypred[i] = self.ytr[min_index]
      return Ypred
  ```
  - 각 테스트 이미지와 train 데이터 비교 -> nearest 한 것 찾아내기
  - NN의 단점 : train 함수의 시간복잡도보다 predict에서 더 오래 걸림
  - Decision Boundary가 Robust 하지 않음

## 3. K-Nearest Neigbor Classifier
- 가까운 이웃을 K개 찾고 그 중 투표를 통해 가장 많은 레이블로 예측
- K값 증가 -> 경계가 부드러워짐 -> Robust
- 이미지를 고차원 공간의 하나의 점이라 생각, 이미지 자체로 생각
- Distance Metric - L2 Distance
  - Euclidean Distance
  - 제곱의 합의 루트를 씌운 값
  - 죄표계와 독립적

## 4. Hyper Parameter
- 우리가 정해줘야 할 거리 척도, K값 등
- 학습되는 것이 아닌 직접 지정
- 학습 데이터의 정확도와 성능을 최대화하는 방법 : 데이터셋 전체 but 한 번도 보지 못한 데이터 예측 문제
- 고안된 방법
  - 전체 데이터셋 -> train셋, test셋
  - train셋으로 학습, test셋 적용 -> 가장 좋은 하이퍼파라미터 선택
  - test셋에서만 잘 동작하는 하이퍼파라미터
- 궁극적인 방법
  - 데이터를 3개의 셋으로 분리
  - train, validation, test
  - train으로 학습, validation으로 검증 -> 가장 좋은 하이퍼파라미터 선택
  - 선택된 classifier로 test셋 단 한 번만 수행
- Cross Validation
  - 데이터가 작을 때 사용
  - 딥러닝에서 거의 사용 X
  - 마지막에 단 한 번 사용할 테스트 데이터 분리
  - 나머지 데이터를 여러 부분으로 분리
  - validation 셋 변경 -> 하이퍼 파라미터 학습 후 최적의 결과 결정

## 5. KNN 정리
- 통계학적 가정 : 데이터는 독립적이며 동일한 분포를 따름
- 일관된 방법론으로 대량의 데이터를 한 번에 수집 -> 무작위로 train, test 분류
- 이미지 분류에 사용 X
- L1, L2 Distance가 이미지 간 거리 척도로써 적절하지 않음
- 고차원의 이미지 -> 모든 공간을 커버할 만큼의 데이터 모으기 불가능

## 6. Linear Classification
- NN과 CNN의 기반이 되는 알고리즘
- Parametric Model : 입력 이미지(X), 파라미터(W, 가중치)
  - train 데이터 정보를 요약해 파라미터에 모아주는 것
  - test 시 더 이상 train 데이터를 직접 비교 X, W만 사용
- Bias : 특정 클래스에 우선권 부여, Scaling Offset 추가
- 클래스의 탬플릿(W)와 인풋 이미지(X)의 유사도 측정
- 고차원 공간의 한 점(이미지) -> 각 클래스를 구분시켜주는 선형 Boundary 역할
- Parity Problem, Multimodal Problem은 어려움