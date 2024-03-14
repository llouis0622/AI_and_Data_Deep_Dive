## 1. Backpropagation(역전파)
- Computational Graph : 연산자와 피연산자들을 각 노드에 배치하여 함수 완성
- 역전파 순서
  - 함수에 대한 Computataional Graph 만들기
  - 각 Local Gradient 구하기
  - Chain Rule
  - z에 대한 최종 Loss L은 이미 계산 -> 최종 목적지는 input에 대한 Gradient 구하기
- Local Gradient 구하기 -> Chain Rule 이용
- 게이트 법칙
  - add gate : 비율만큼 나눠줌
  - max gate : 하나만 실제로 영향을 주는 값, 그쪽만 Gradient를 가짐
  - 하나의 노드에서 Forward로 Multi 나가면 역전파할 때 Gradient 더해줌
- input이 벡터 -> Jacobian 계산
  - 입력의 각 요소는 출력의 해당 요소에만 영향 -> 대각행렬
- 벡터의 Gradient : 항상 원본 벡터의 사이즈와 동일
- 곱셈 게이트에 대한 Forward와 Backward Pass
  ```
  class MultiplyGate(object):
    def forward(x, y):
      z = x * y
      self.x = x
      self.y = y
    
    def backward(dz):
      dx = self.y * dz
      dy = self.x * dz
      return [dx, dy]
  ```

## 2. Neural Network(신경망)
- Single 변환이 아닌 그 이상의 레이어를 쌓음
- max와 같은 비선형 레이어 추가 가능
- input layer - hidden layer ... - output layer
- Activation Functions
  - Sigmoid
  - Leaky ReLU
  - tanh
  - Maxout
  - ReLU
  - ELU