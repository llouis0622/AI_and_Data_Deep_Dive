# 1. 통계학과 머신러닝의 관계

- 다양한 알고리즘 개념과 수식 → 통계학 적용 가능

# 2. 확률 변수와 확률 분포

## 1. 확률 변수

- 확률(Probability) : 어떤 사건이 일어날 가능성을 수치화시킨 것
- 모든 확률이 0과 1 사이에 있음
- 발생 가능한 모든 사건의 확률을 더하면 1이 됨
- 동시에 발생할 수 없는 사건들에 대해 각 사건의 합의 확률은 개별 확률이 일어날 확률의 합과 같음
- 확률 변수(Random Variable) : 결과값이 확률적으로 정해지는 변수

## 2. 확률 분포(Probability Distribution)

- 확률 변수가 특정값을 가질 확률의 함수
- 확률 변수가 특정값을 가질 확률이 얼마나 되느냐를 나타내는 것
- 이산 확률 변수(Discrete Random Variable) : 확률 변수가 가질 수 있는 값
- 이산 확률 분포(Discrete Probability Distribution) : 이산 확률 변수의 확률 분포
- 확률 질량 함수(Probability Mass Function, PMF) : 이산 확률 변수에서 특정값에 대한 확률을 나타내는 함수
- 연속 확률 변수(Continuous Random Variable) : 확률 변수가 가질 수 있는 값의 개수를 셀 수 없는 것
- 연속 확률 분포(Continuous Probability Distribution) : 연속 확률 변수의 확률 분포
- 확률 밀도 함수(Probability Density Function, PDF) : 연속 확률 변수의 분포를 나타내는 함수
- 누적 분포 함수(Cumulative Distribution Function, CDF) : 주어진 확률 변수가 특정값보다 작거나 같은 확률을 나타내는 함수
- 결합 확률 밀도 함수(Joint Probability Density Function) : 확률 변수 여러 개를 함께 고려하는 확률 분포
- 독립 항등 분포(Independent and Identically Distributed, IID) : 두 개 이상의 확률 변수를 고려할 때 각 확률 변수가 통계적으로 독립이고 동일한 확률 분포를 따르는 것

# 3. 모집단(Population)과 표본(Sample)

- 모집단 : 관심이 있는 대상 전체
- 표본 : 모집단에서 일부를 추출한 것
- 모수(Population Parameter) : 모집단의 특성을 나타내는 대표값
- 표본 통계량(Sample Statistic) : 표본의 대표값

# 4. 평균과 분산

## 1. 평균

- 모든 데이터값을 덧셈한 후 데이터 개수로 나누는 것
- 모평균(Population Mean) : 모집단의 평균
- 표본 평균(Sample Mean) : 모평균의 추정량

## 2. 분산

- 데이터가 얼마나 퍼져 있는지를 수치화한 것
- 평균에 대한 편차 제곱의 평균으로 계산
- 모분산(Population Variance), 표본 분산(Sample Variance)
- 표준 편차 : 분산의 양의 제곱근
- 평균 : 데이터의 중심 표현
- 분산 : 데이터의 흩어짐 정도 표현

## 3. 평균과 분산의 성질

- 서로 독립인 확률 변수의 합에 대한 기댓값