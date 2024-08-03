# 1. 확률(Probability)

- 어떤 사건이 일어날 것이라고 믿는 정도
- 아직 일어나지 않은 사건에 대한 예측 정량화
- 가능도(Likelihood) : 이미 발생한 사건의 빈도 측정

## 1. 확률 대 통계

- 확률 : 어떤 사건이 일어날 가능성에 대한 이론적 개념, 데이터 불필요
- 통계 : 데이터 없이 존재 불가, 데이터를 사용해 확률 발견, 데이터 설명 도구 제공

# 2. 확률 계산

- 주변 확률(Marginal Probability) : 사건의 단일 확률을 다룰 때의 확률

## 1. 결합 확률(Joint Probability)

- 두 개의 사건에 대한 개별 확률이지만 두 사건이 함께 발생할 확률
- 곱셈 정리

## 2. 합 확률(Union Probability)

- 사건이 따로 발생할 확률
- 상호 배타적 사건
- 덧셈 법칙

## 3. 조건부 확률과 베이즈 정리

- 조건부 확률(Conditional Probability) : 사건 B가 발생했을 때 사건 A가 발생할 확률
- 베이즈 정리(Bayes’ Theorem) : 조건부 확률에서의 조건을 뒤집는 것

## 4. 결합 및 합 조건부 확률

- `P(A AND B) = P(A|B) * P(B)`
- `P(A OR B) = P(A) + P(B) - P(A|B) * P(B)`

# 3. 이항 분포(Binomial Distribution)

- 확률이 p일 때 n번의 시도 중 k번이 성공할 가능성

# 4. 베타 분포(Beta Distribution)

- 알파 번의 성공과 베타 번의 실패가 주어졌을 때 사건이 발생할 수 있는 다양한 기본 확률의 가능성
- 누적 분포 함수(CDF, Cumulative Distribution Function)
- 제한된 표본 집합 → 사건이 발생할 확률과 발생하지 않을 확률 측정 가능