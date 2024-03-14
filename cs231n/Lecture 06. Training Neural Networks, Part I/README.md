## 1. Activation Functions
- input -> 가중치와 곱해짐, 비선형 함수인 활성함수를 거쳐 해당 데이터의 활성화 여부 결정
- Sigmoid
  - 음/양의 큰 값에서 Saturation 되는 것이 Gradient를 없앰
  - x가 0에 가까운 것은 잘 동작
  - 출력이 Zero Centered 아님
  - x가 항상 양수 -> W의 Gradient는 시그모이드 Upstream Gradient 부호와 항상 동일
  - 양 또는 음의 방향으로 밖에 업데이트 하지 못함 -> Zig Zag Path, 비효율적
  - `exp()` 연산 Cost 비쌈
- Tanh
  - Zero-Centered 문제 해결
  - Saturation 문제 존재
- ReLU
  - x 양수 -> Not Saturation
  - 계산 효율 우수
  - 생물학적 타당성 존재
  - Zero-Centered 아님
  - 음의 영역 Saturation
  - Gradient 절반 죽임 -> Dead ReLU
  - 초기화 시 Positive Biases 추가 -> Active 확률 높이기
  - Zero-bias 사용
- Leaky ReLU
  - Zero-Mean
  - 음의 영역에서도 Saturation 되지 않음
  - Dead ReLU 없음
- PReLU
  - Leaky ReLU와 비슷
  - 기울기 Alpha로 결정
- ELU
  - Zero-Mean에 가까운 출력값
  - 음에서 Saturation, 노이즈에 강인함
- Maxout Neuron
  - 기본 형식 미지정
  - 두 개의 선형함수 중 큰 값 -> ReLU와 Leaky ReLU의 일반화 버전
  - 선형 -> Not Saturation & Gradient 죽지 않음
  - W1, W2 -> 파라미터 2배

## 2. Data Processing
- Zero-Mean으로 만들고 표준편차로 Normalize
- 이미지 : Zero-Mean만 함
- 입력이 전부 positive한 경우 방지 -> 학습최적화

## 3. Weight Initialization
- 모든 파라미터 0 설정
  - 모든 뉴런이 같은 일을 함
  - 모든 가중치가 똑같은 값으로 업데이트
  - 가중치 동일 -> Symmetry Breaking
- 임의의 작은 값으로 초기화
- 초기 W를 표준정규분포에서 샘플링
- Xavier Initialization
  - Standard Gaussian으로 뽑은 값을 입력의 수로 스케일링
  - 입출력의 분산 맞춰줌
  - 입력 수가 작으면 더 작은 값으로 나누고 좀 더 큰 값을 얻음
  - 각 레이어의 입력 -> Unit Gaussian

## 4. Batch Normalization
- 레이어의 출력 -> Unit Gaussian
- 현재 Batch에서 계산한 Mean과 Varience를 이용해 Normalization
- 학습동안 모든 레이어의 입력 -> Unit Gaussian
- FC or Conv Layer 직후에 넣음
- Conv에서 같은 Activation Map의 같은 채널에 있는 요소들은 같이 Normalization 함

## 5. Babysitting the Learning Process
- 데이터 전처리
- 아키텍쳐 선택
- 네트워크 초기화
- 초기 Loss 체크 : Soft Max 로그 체크 -> Regularization Term 추가 후 Loss 증가 체크
- 데이터 일부만 학습 : Regularization 사용 X, Epoch 마다 Loss 내려가는지 확인, Train Accuracy 증가 확인
- Learning Rate : Regularization 후 Learning Rate 찾기, 작으면 Loss 감소 X, 크면 NaNs Cost 발산, e-3, e-5 사이 사용

## 6. HyperParameter Optimization
- Cross-Validation : Training Set으로 학습, Validation Set으로 평가
- Coarse Stage : Epoch 몇 번으로 좋은지 아닌지 판단 -> 범위 결정
  - 로그 스페이스에서 차수 값만 샘플링
- Fine Stage : 학습 길게 함
  - train 동안 Cost 변화
  - reg 범위, lr 범위 정함
  - 최적 값이 범위의 중앙 쯤에 위치하도록 범위 설정
  - Random Search -> Important Variable에서 다양한 값 샘플링
- Loss Curve
  - 평평하다가 갑자기 가파르게 내려감 -> 초기화 문제
  - Gradient 역전파 초기에는 잘 되지 않다가 학습이 진행되면서 회복
  - train & Va Accuracy 큰 차이 -> 오버핏 -> Regularization 강도 높이기
  - Gap 미존재 -> Not Overfit, Capacity 증가 가능
- 가중치 크기 : 파라미터의 Norm 구하기