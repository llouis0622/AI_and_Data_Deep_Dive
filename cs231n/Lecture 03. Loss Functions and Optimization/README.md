## 1. Loss Fuction(손실 함수)
- Multiclass SVM Loss
  - Hinge Loss
  - 0 if syi(정답 카테고리의 스코어) >= sj(오답 카테고리의 스코어) + 1 else sj - syi + 1
  - 마진을 크게 -> 널널하게 분류, 작게 -> 엄격하게 분류
  - Squared Hinge Loss : 큰 Loss를 더 크게, 아주 작은 Loss는 더 작게
    ```
    def L_i_vectorized(x, y, W):
      scores = W.dot(x)
      margins = np.maximum(0, scores - scores[y] + 1)
      margins[y] = 0
      loss_i = np.sum(margins)
      return loss_i
    ```

## 2. Regularization Loss
- Loss가 작으면 작을수록 좋은것인가 -> train 데이터에 완벽한 W, test 데이터의 성능 상관 X
- 함수가 단순해야 test 데이터를 맞출 가능성 증가
- Data Loss + Regularization, 하이퍼 파라미터인 람다로 두 항간의 트레이드 오프 조절
- Regularization
  - L2 Regularization
  - L1 Regularization
  - Elastic Net(L1 + L2)
  - Max Norm Regularization
  - Dropout
  - Fancier:Batch Normalization, Stochastic Depth
- L2 vs L1
  - L2 - Norm 작음
  - L2 - Coarse 한 것을 고름
  - L2 - 모든 요소가 골고루 영향을 미치길 바람
  - L2 - Parameter Vector, Gaussian Prior, MAP Inference
  - L1 - Sparse한 Solution을 고름
  - L1 - 0이 많으면 좋다고 판단

## 3. Multinomial Logistic Regression(Softmax)
- 스코어 자체에 추가적인 의미 부여
- 클래스별 확률 분포 계산 -> Loss 계산
- 스코어 자체를 Loss로 쓰는 것이 아닌 지수화 -> 정규화 -> -log() 사용

## 4. Optimization(최적화)
- 최종 손실함수가 최소가 되게 하는 W를 구하는 것
- 임의의 지점에서 시작해서 점차 성능 향상 : Iterative 방식
- 해당 지점에서의 Slope 계산 -> 낮은쪽으로 이동(Gradient 사용)
- Gradient
  - 벡터 X의 각 요소의 편도함수들의 집합
  - 어떤 방향으로 갈 때 함수의 Slope가 어떤지를 알려줌
  - 방향의 경사 파악 -> 방향의 Unit Vec, Gradient 내적
  - 수치적 Gradient : 유닛 테스트로 사용
  - 해석적 Gradient : 실제 구현, Loss와 Gradient 계산 후 가중치를 Gradient 반대 방향으로 업데이트
- Gradient Descent : Momentum, Adam Optimizer
- Stochastic Gradient Descent
  - 전체 데이터 셋의 Gradient, Loss 계산 X -> Minibatch라는 작은 샘플 집합으로 나눠 학습 진행
  - 32, 64, 128 사용
  - Loss와 Gradient 추정치 구함 -> Monte Carlo Method와 유사
- 특징 벡터 사용 -> Linear Classifier의 input -> 극좌표계로 특징 변환 -> 분류 가능