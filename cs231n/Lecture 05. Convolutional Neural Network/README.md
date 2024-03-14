## 1. NN의 역사

### 신경망
- 1957년
  - Frank Rosenblatt - Mark I Perceptron Machine
  - 최초의 퍼셉트론 기계
- 1960년
  - Widrow & Hoff - Adaline and Madaline
  - 최초의 Multilayer Perceptron Network
- 1986년
  - Rumelhart
  - 역전파 제안, 신경망 학습 시작
- 2006년
  - Geoff Hinton & Ruslan Salakhutdinow
  - DNN 학습가능성
- 2012년
  - Hintin lab
  - 음성인식 NN 성능 좋음
  - ImageNet Classification에서 최초로 NN 사용, 결과 우수 -> AlexNet

### CNN
- 1950년
  - Hubel & Wiesel - 뉴런 연구
  - Topographical Mapping
  - 뉴런의 계층구조
  - Simple Cells -> Complex Cells -> Hypercomplex Cells
  - 빛의 방향 -> 움직임 -> 끝 점
- 1980년
  - Necognitron
  - Simple/Complex Cell 사용한 최초의 NN
- 1998년
  - Yann LeCun
  - NN 학습을 위한 역전파의 Gradient-based Learning
- 2012년
  - Alex Krizhevsky
  - CNN의 유행

## 2. CNN 활용
- 이미지 분류
- Detection
- Segmentation
- 자율주행
- 얼굴인식 사람추정
- 자세 인식
- 의학 진단
- Image Captioning
- 화풍 변경

## 3. CNN 원리
- Convoluional Layer : 기존의 Structure 보존하며 계산
  - 이미지 안을 슬라이딩하면서 공간적인 내적
  - 모든 Depth에 대해 내적 -> 필터의 Depth는 input의 Depth와 동일
- Filter를 겹쳐놓고 내적 -> 슬라이딩해서 옆에서 계속 내적 -> output activation map의 해당 위치에 전달
- Activation Map의 차원
- Convolution Layer : 여러 개 필터 사용 -> 필터마다 다른 특징 추출 가능
- 자신이 원하는 만큼 필터 사용 가능 -> 필터 사이에 activation, pooling 도입
- 여러 개의 필터를 가지고 각 필터마다 각각의 출력 Map -> 각 필터들이 계층적으로 학습
- 여러 개의 Conv Layer를 거치면서 단순한 구조에서 복잡한 구조로 찾아감
- CNN 과정
  - input 이미지 -> 여러 레이어를 거침
  - FC Layer를 통해 스코어 계산
  - 필터를 몇 칸씩 움직일지 Stride로 정함
  - input 사이즈와 슬라이딩 시 딱 맞아 떨어지는 Stride 이용
  - Output Size : (N(입력의 차원) - F(필터 사이즈)) / stride + 1
- stride 설정 -> 다운샘플링 가능, 성능 우수
- activation map 사이즈 감소 -> FC Layer 파라미터 수 줄어듦

## 4. Zero-Padding
- 코너의 값들이 적게 연산되는 것을 막아주고 레이어들을 거치면서 입력의 사이즈가 줄어드는 것을 막아줌
- 깊은 네트워크 : Activation Map이 엄청 작아짐 -> 정보를 잃음
- 필터 사용법
  - 3 * 3 필터 stride = 1
  - 5 * 5 필터 stride = 2
  - 7 * 7 필터 stride = 3
  - 필터 개수 : 2의 제곱수(32, 64, 128, 512)
- 필터 : 가중치
- Receptive Field : 한 뉴런이 한 번에 수용할 수 있는 영역

## 5. Pooling, ReLU
- Pooling Layer
  - Representation을 더 작고 관리하기 쉽게 해줌
  - DownSample
  - 공간적 Invariance
  - Depth는 그대로 둠
  - 차원 계산 : (Width - Filter) / Stride + 1
  - 보통 Padding 안함
  - 2 * 2, 3 * 3, stride = 2
- Max Pooling
  - 필터 크기와 stride 정하면 됨
  - 필터 안에 가장 큰 값 고름
  - 겹치지 않게 풀링
- ReLU Layer
  - 실제 방식과 가장 유사한 비선형함수
  - 활성화, 비활성화 결정