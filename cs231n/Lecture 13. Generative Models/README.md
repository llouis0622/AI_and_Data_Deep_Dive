## 1. Unsupervised Learning
- 데이터가 label 없이 주어졌을 때 데이터의 숨어있는 기본적인 구조를 학습하는 것
- Clustering
- Dimensionality Reduction
- Feature Learning
- Density Estimation
- 데이터를 많이 모을 수 있음 but 확실한 학습 결과로 분류 기준을 사람이 알 수 없음

## 2. Generative Models
- training data -> 동일한 분포에서 새로운 샘플들을 생성하는 것
- Density Estimation
  - Explicit Density Estimation : 생성 모델을 명시적으로 나타냄
  - Implicit Density Estimation : 생성 모델을 정의하지 않고 구함
- Sample for Artwork
- Super-Resolution
- Colorization

## 3. PixelRNN/CNN
- Explicit Density Estimation 중 계산 가능한 Density 다룸
- Chain Rule -> Image를 1차원 분포들 간의 곱 형태로 분해
- 이전 픽셀 정의
- 인접한 픽셀들을 이용하여 순차적으로 픽셀 생성 -> LSTM으로 모델링
- 특정 영역만 사용 -> 빠르게 이미지 생성 가능

## 4. Variational Autoencoders
- 확률 모델이 계산 불가능한 함수로 정의
- 하한선을 구해 계산 가능한 형태로 변환
- 웝본 재복원 -> 학습
- Autoencoder -> 중요한 특징 벡터들을 잘 추출 -> 벡터들을 이용하여 이미지 재구성
- 학습 데이터로 관측할 수 없는 잠재 변수에 대해 재구성

## 5. GAN(Generative Adversarial Networks)
- 확률 분포를 계산하는 것을 포기하고 단지 샘플만 얻어내는 방법
- 결과만 뽑아냄
- Generator Network : Discriminator를 속여 실제처럼 보이는 가짜 이미지 생성
- Discriminator Network : 진짜 이미지와 가짜 이미지 구별
- Gradient 정도가 오차가 심할 때 크지 않음
- DCGAN : CNN + GAN
  - 진짜 같은 이미지 만들어냄
  - 해석 가능한 벡터들을 가지고 연산 수행 -> 새로운 이미지 생성
  - 