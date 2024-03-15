## 1. RNN
- Sequence를 따라 node 사이의 연결의 형태가 방향 그래프인 인공 신경망
- one to one : 가장 기본적인 형태, 하나의 입력, 하나의 출력
- one to many : 하나의 입력에 대해 여려 출력
- many to one : 여러 입력에 대해 하나의 출력
- many to many : 여러 입력에 대해 여러 출력, 입력에 대해 바로 출력 가능 or 이후에 나오기도 함

## 2. RNN의 기본 구조와 수식
- t : 시간
- W : weight 값
- ht = fw(ht-1, xt)
- hidden state의 값에 대한 Whh와 현재 입력값에 대한 Wh weight 값 곱함

## 3. RNN Computational Graph
- RNN을 순차적으로 쌓아 올려, 여러 입력 받기 가능
- 각 hidden state에서 다른 네트워크에 들어가 yt 나타냄
- 같은 weight 값 사용
- 최종 hidden state에서만 결과 값이 나옴
- Seq2Seq Model
  - 기계번역에서 사용
  - Mony to one 과 ono to many의 조합
  - Encoder + Decoder

## 4. Truncated Backpropagation
- 배치별로 나누어서 학습 진행
- 배치 크기만큼에 loss를 보고 학습 진행
- Tensorflow -> batch size만큼 뛰어가면서 하는 것도 가능

## 5. Image Captioning
- CNN에서 나오는 하나의 출력 값을 RNN의 입력으로 사용 -> 문장 만들어냄
- Supervised Learning
- Attention : Caption을 생성할 때 이미지의 다양한 부분 집중 가능
  - 이미지의 모든 부분으로부터 단어 뽑아냄 -> 디테일 신경 가능
  - Bottom-Up Approach
  - 각 벡터가 공간정보를 가지고 있는 Grid of Vector 제작
- Visual Question Answering : 사진에 대한 질문과 답 제작

## 6. Multilayer RNN
- hidden layer를 하나만 사용하는 것이 아닌 여러 개 사용

## 7. Backpropagation Through Time(BPTT)
- 뒤에서부터 역전파 시작
- h4에서 부터 시작하여 h0까지의 loss -> W의 transpose 요소를 모두 곱해야하는 비효율적인 연산이 반복
- Exploding gradients : Gradient clipping 으로 해결
- Vanishing gradients : RNN의 구조를 변경하여 해결(LSTM)
- 장기 의존성(Long-Term Dependency) : 오래전에 입력에 정보를 사용하지 못하는 문제을 구조 변경(LSTM)

## 8. LSTM(Long Short Term Memory)
- i : input gate @sigmoid, 현재 입력 값을 얼마나 반영할 것인지
- f : forget gate @sigmoid, 이전 입력 값을 얼마나 기억할 것인지
- o : output gate @sigmoid, 현재 셀 안에서의 값을 얼마나 보여줄 것인지
- g : gate gate @tanh, input cell을 얼마나 포함시킬지 결정하는 가중치, 얼마나 학습시킬지 (-1, 1)
- 장점
  - forget gate의 elementwise multiplication이 matrix multiplication보다 계산적 효율성을 가짐
  - forget gate 값을 곱하여 사용하므로 항상 같은 weight값을 곱해주던 위 형태와 다르게 입력에 따라 다른 값을 곱해주어 exploding 또는 vanishing 문제를 피하는 이점
  - forget gate의 sigmoid 값으로 bias를 1에 가깝게 만들어주면 vanishing gradient를 많이 약화
  - Gradient를 구하기 위해서 W 값이 곱해지지 않아되기 때문에 마치 고속도로 처럼 gradient를 위한 빠른 처리
  - ReNet의 residual block과 비슷

## 9. GRU(Gated Recurrent Unit)
- LSTM에서 ct와 ht가 하나의 ht
- rt의 추가로 과거의 정보를 어느정도 reset 할지 정함
- Update를 위해 사용되던 f와 i의 값이 zt와 (1-zt)인 하나 값으로 input과 hidden state의 update 정도 정함
- 현 시점의 정보 후보군(Candidate)을 계산합니다. gt는 과거 hidden state(은닉층) 값을 그대로 사용하지 않고 reset gate(rt)를 곱함
- 현 시점 hidden state(은닉층) 값은 update gate 결과와 Candidate 결과를 결합
- 일반적으로 LSTM과 GRU은 거의 비슷한 성능
- GRU가 LSTM보다 단순한 구조를 가져 학습이 더 빠름
- 하이퍼파라미터를 더 빨리 찾은 모델을 사용