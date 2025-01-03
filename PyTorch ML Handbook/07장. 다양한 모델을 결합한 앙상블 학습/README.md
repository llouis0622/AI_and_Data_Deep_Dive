# 1. 앙상블 학습(Ensemble Learning)

- 여러 분류기를 하나의 메타 분류기로 연결하여 개별 분류기보다 더 좋은 일반화 성능을 달성하는 것
- 과반수 투표(Majority Voting) 방식

# 2. 다수결 투표를 사용한 분류 앙상블

## 1. 간단한 다수결 투표 분류기 구현

- 코드 작성

## 2. 다수결 투표 방식을 사용하여 예측 만들기

- 코드 작성

## 3. 앙상블 분류기의 평가와 튜닝

- 코드 작성

# 3. 배깅 : 부트스트랩 샘플링을 통한 분류 앙상블

## 1. 배깅 알고리즘의 작동 방식

- 원본 훈련 데이터셋에서 부트스크랩 샘플을 뽑아서 사용
- 배깅 단계마다 중복을 허용하여 랜덤하게 샘플링

## 2. 배깅으로 Wine 데이터셋의 샘플 분류

- 코드 작성

# 4. 약한 학습기를 이용한 에이다부스트(AdaBoost)

- 부스팅 : 분류하기 어려운 훈련 샘플에 초점
- 잘못 분류된 훈련 샘플을 그 다음 약한 학습기가 학습하여 앙상블 성능 향상

## 1. 부스팅 작동 원리

- 훈련 데이터셋에서 중복을 허용하지 않고 랜덤한 부분 집합을 뽑아 약한 학습기 훈련
- 훈련 데이터셋에서 중복을 허용하지 않고 두 번째 랜덤한 훈련 부분 집합을 뽑고 이전에 잘못 분류된 샘플의 50%를 더해 약한 학습기 훈련
- 훈련 데이터셋에서 앞 두개의 학습기에서 잘못 분류한 훈련 샘플을 찾아 세 번째 약한 학습기 훈련
- 약한 학습기를 다수결 투표로 연결

## 2. 사이킷런에서 에이다부스트 사용

- 코드 작성

# 5. 그레이디언트 부스팅(Gradient Boosting) : 손실 그레이디언트 기반의 앙상블 훈련

## 1. 에이다부스트와 그레이디언트 부스팅 비교

- 에이다부스트 : 이전 트리 오차를 기반으로 깊이가 1인 결정 트리 훈련
- 그레이디언트 부스팅 : 예측 오차를 사용하여 반복적인 스타일로 결정 트리 훈련

## 2. 그레이디언트 부스팅 알고리즘 소개

- 일련의 트리 생성
- 각 트리 → 이전 트리 오차에서 훈련
- 상수 예측 값 반환 모델 초기화
- 예측값과 클래스 레이블 사이 차이 계산
- 의사 잔차에서 트리 훈련
- 출력값을 이전 트리에 더해 모델 업데이트

## 3. 분류를 위한 그레이디언트 부스팅 알고리즘

- 코드 작성

## 4. 그레이디언트 부스팅 분류 예제

- 코드 작성

## 5. XGBoost 사용하기

- 몇 가지 트릭과 근사 방법 도입 → 훈련 속도 증가