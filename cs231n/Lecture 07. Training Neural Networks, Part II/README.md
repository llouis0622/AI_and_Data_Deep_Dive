## 1. Fancier Optimization
- SGD 단점
  - 가중치마다 변하는 속도가 다름 -> 지그재그 형태로 업데이트 진행
  - High Condition Number : 해당 지점의 Hessian Matrix 최대, 최소 Singluar Values 값의 비율이 너무 큼
  - Local Minima : 중간 Valley에서 Gradient가 0이 되어 업데이트 중지
  - Saddle Points 더 빈번하게 발생
- Momentum Term
  - Gradient 계산할 때 Velocity 이용
  - 떨어지면서 속도 빨라짐, Gradient 0이 되어도 업데이트 가능
- Nesterov Accelerate Gradient
  - Velocity의 방향이 잘못되었을 경우 현재 Gradient로 보정
  - Convex에서는 뛰어남, Non-Convex에서는 성능 떨어짐
  - 평평한 Minima가 더 Robust
- AddGrad
  - Velocity -> Grad Squared Term 사용
  - Small Dimension : 제곱합이 작으므로 가속
  - Large Dimension : 속도 감소
  - 진행 시 Step 줄어듦 -> Non Convex에서 Saddle 걸리면 멈춤
- RMSProp
  - Decay Rate(0.9, 0.99) 곱셈
- Adam
  - Momentum + Ada
  - First Moment : Gradient 가중 합 -> Velocity 담당 -> Momentum 역할
  - Second Moment : 제곱 -> 나눠줌, 속도 조절 -> 제곱항 역할
  - Second Moment 0으로 초기화 -> Step 증가 -> Bias Correction Term 추가 보정
  - beta1 = 0.9, beta2 = 0.999, 학습률 e-3, e-4 -> 거의 모든 아키텍쳐에서 잘 동작함
  - 각 차원마다 속도 조절 가능 but 언제나 축방향으로만 가능

## 2. Learning Rate
- 수렴을 잘 하고 있는데 Gradient 작아짐 -> Learning Rate가 너무 높아서 깊게 들어가지 못함
- First-Order Optimization
- Second-Order Optimization
  - 2차 테일러 근사함수 -> Hessian 역행렬 -> Minima로 곧바로 이동
  - 근사함수 제작 -> 그 최소값으로 이동
  - Quasi Newton Method -> Low Rank 근사
  - L-DFGS : Stocahstic Case, Non Convex에서 안 좋음

## 3. Model Ensembles
- 10개의 모델을 독립적으로 학습 -> 결과의 평균 이용 -> 성능 2% 향상
- 학습 도중 모델 저장, 앙상블로 사용
- Learning Rate를 낮췄다 높였다 반복 -> 손실함수가 다양한 지역에 수렴할 수 있도록 함

## 4. Regularization
- 앙상블 말고 단일 모델 성능 증가
- Dropout
  - Forward Pass 과정에서 일부 뉴런을 0으로 만듦
  - 한 레이어의 출력을 전부 구한 후 일부를 0으로 만듦
  - FC나 Conv에서 사용
  - 네트워크가 특정 Feature에만 의존하지 못하게 함
  - 단일 모델로 앙상블 효과
  - test time : dropout probability를 출력에 곱해서 train과 기대값을 같게 만듦
  - train에서 p를 나눠서 사용
  - 각 스텝마다 업데이트되는 파라미터 수 줄어듦 -> 학습시간 증가 but 일반화능력 우수
  - bn과 유사
  - train time : stochasticity 추가
  - test time : marginalize out
- Dropconnect
  - weight matrix를 임의로 0으로 만듦
- Fractional Max Pooling : Pooling 연산을 어디서할지 랜덤 선정
- Stochastic Depth
  - train time에 네트워크 레이러를 랜덤 드롭, 일부만 사용해서 학습
  - overfit 조짐 -> 하나씩 추가, BN, Dropout 사용

## 5. Data Augmentation
- Flip
- Crop
- Color Jittering

## 6. Transfer Learning
- 모델 -> 빠르게 학습시키기
- CNN을 가지고 ImageNet 같은 아주 큰 데이텃세으로 학습 한 번 진행
- 마지막 FC Layer -> 지금 필요한 것으로 변경
- 마지막 레이어 -> 데이터 학습 가능
- 학습률 낮춰서 사용
- 가중치 수정 최소화
- 유사 데이터셋으로 학습된 Pretrained Model -> Fine Tune