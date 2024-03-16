## 1. Reinforcement Learning
- 머신러닝의 한 범주
- 어떤 환경 안에서 정의된 에이전트가 현재의 상태 인식 -> 선택 가능한 행동 중 보상을 최대화하는 행동을 선택
  - 에이전트 : 환경 내애서 행동을 취할 수 있는 물체 -> 최대로 받는 것이 목표
  - 환경 : 에이전트와 상호작용, 적절한 상태 부여
- Cart-Pole Problem
- Robot Locomotion
- Atari Games
- GO

## 2. Markov Decision Processes
- 강화 학습 방법 수식화
- 가능한 상태 집합, 액션 집합 들로 보상을 구하는 것
- 누적 보상 최대화하는 파이값 찾기
- 미래 보상들의 합에 대한 기대값을 올림

## 3. Q-Learning
- NN으로 근사시키는 것
- Bellman Equation 만족
- 한 번의 Forward Pass만으로 모든 함수에 대한 Q-Value 값 계산 가능

## 4. Policy Gradient
- Fuction 복잡함 -> state 차원 증가
- 정책 자체 학습
- Gradient Ascent -> 파라미터 값 업데이트
- 경로에 대한 미래 보상 기대값 나타냄
- 구체적인 값이 없이도 정책 자체의 그레이디언트로 최적의 정책 구하기 가능