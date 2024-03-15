## 1. ConvNet
- First Layer : 필터 시각화
  - 이미지 내의 edge 특징 뽑아냄
  - 필터 깊어짐 -> 합성곱 -> 복잡해짐
  - 가중치들이 내적 -> 층 깊어짐 -> 직관적으로 이해할 수 없는 필터
- Last Layer : 특징 벡터들을 Nerest Neighbor -> 이미지 카텓고리
  - t-SNE, PCA로 2차원 군집화
- Feature Map -> 직관적으로 이해할 수 있는 것들이 일부

## 2. Maximally Activating Patches
- 한 부분을 집중해서 Layer가 깊어질수록 계속해서 그 부분에 집중
- 한 채널에서 모든 뉴런은 필터를 거쳐 같은 가중치를 가짐

## 3. Saliency Maps
- 슬라이드처럼 이미지를 객체의 형상으로 제작

## 4. Intermediate Features
- 이미지 픽셀에 관련된 뉴런의 기울기 구함
- Positive Gradient만 살려서 Gradient 값 사용
- Negative 부분 -> ReLU 함수에서 전부 0, 양수 사용
- 뉴런의 중요한 부분 작동 -> 디테일하게 해소 가능
- 어떤 문구에 집중 -> >뉴런들이 가로, 세로의 line, edges에 집중

## 5. Visualizing CNN
- Gradient Ascent : 이미 학습이 된 네트워크의 가중치들을 전부 고정 -> Score가 최대가 되는 방향으로 백지를 넣음
  - synthetic 이미지 제작
  - 특정 뉴런의 값을 최대화시키는 방향으로 업데이트
  - 특정 뉴런들에만 오버피팅 되는 문제 방지
  - 이미지 픽셀을 전부 0값으로 초기화 -> 백지
  - Forward를 통해 이미 가중치 고정 네트워크에 넣어 스코어 계산
  - image pixel 뉴런 값들의 gradient -> backprop 구함
  - 이미지를 특정 뉴런들의 최대화를 위해 픽셀 단위로 업데이트 진행

## 6. DeepDream
- 중간의 Layer에서 모든 뉴런 값들을 확대함
- Forward 과정에서 선택했던 Layer의 Activation 값 계산
- Activation 값들로 Gradient set
- Backward를 통해 그대로 학습
- 반복하여 이미지 제작
- Layer가 깊으면 깊을수록 학습 완성도 증가

## 7. Feature Inversion
- Feature Map으로 이미지의 특징들을 통해 이미지 재구성
- 원본 이미지를 재생성
- 새로 생성된 Feature Map의 Vector와 원래의 Image Feature Vector 간의 distance 최소화
- Layer 깊어짐 -> 정확한 픽셀, 컬러, 텍스쳐 버림 -> 전체적인 구조에 집중

## 8. Neural Texture Synthesis
- 특정 레이어 -> 레이어가 깊을수록 더 잘 표현됨
- Neural Style Transfer
- style image : gram matrix -> texture
- content image : feature inversion -> feature map -> 디테일이 떨어진 이미지 생성