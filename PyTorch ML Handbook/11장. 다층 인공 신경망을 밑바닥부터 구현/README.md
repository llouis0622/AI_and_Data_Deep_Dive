# 1. 인공 신경망으로 복잡한 함수 모델링

## 1. 단일층 신경망 요약

- 아달린(Adaline, Adaptive Linear Neuron) : 이진 분류 수행
- 경사 하강법 최적화 알고리즘 → 모델 가중치 학습
- 전체 훈련 데이터셋에 대한 그레이디언트 계산 → 그레이디언트 반대 방향으로 진행하도록 가중치 업데이트

## 2. 다층 신경망 구조

- 여러 개의 단일 뉴런 연결 → 다층 피드포워드 신경망
- 다층 퍼셉트론(MLP, MultiLayer Perceptron) : 데이터 입력, 하나의 은닉층, 하나의 출력층
- 심층 신경망(Deep Neural Network) : 하나 이상의 은닉층을 가진 네트워크

## 3. 정방향 계산으로 신경망 활성화 출력 계산

- 정방향 계산(Forward Propagation)
    - 입력층 → 정방향으로 훈련 데이터 패턴 네트워크에 전파
    - 네트워크 출력 기반 → 비용 함수를 이용하여 최소화해야 할 오차 계산
    - 네트워크에 있는 모든 가중치에 대한 도함수 → 오차 역전파, 모델 업데이트
- 피드포워드 : 각 층에서 입력을 순환시키지 않고 다음 층으로 전달하는 것

# 2. 손글씨 숫자 분류

## 1. MNIST 데이터셋 구하기

- 코드 작성

## 2. 다층 퍼셉트론 구현

- 역전파(Backpropagation) : 가중치 및 절편 파라미터에 대한 손실의 그레이디언트 계산

## 3. 신경망 훈련 루프 코딩

- 코드 작성

## 4. 신경망 모델의 성능 평가

- 코드 작성