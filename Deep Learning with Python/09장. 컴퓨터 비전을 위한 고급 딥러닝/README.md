# 1. 세 가지 주요 컴퓨터 비전 작업

- 이미지 분류(Image Classification) : 이미지에 하나 이상의 레이블 할당
- 이미지 분할(Image Segmentation) : 이미지를 다른 영역으로 나누거나 분할
- 객체 탐지(Object Detection) : 이미지에 있는 관심 객체 주변에 사각형을 그리는 것
- 이미지 유사도 평가(Image Similarity Scoring) : 두 이미지가 시각적으로 얼마나 비슷한지 추정
- 키포인트 감지(Keypoint Detection) : 이미지에서 관심 속성 정확히 짚어 내기
- 포즈 추정(Pose Estimation)
- 3D 메시 추정(3D Mesh Estimation)

# 2. 이미지 분할 예제

- 이미지 분할 : 이미지 안의 각 픽셀에 클래스를 할당하는 것
    - 시맨틱 분할(Semantic Segmentation) : 각 픽셀이 독립적으로 하나의 의미를 가진 범주로 분류
    - 인스턴스 분할(Instance Segmentation) : 이미지 픽셀을 범주 분류 + 개별 객체 인스턴스 분류
- 분할 마스크(Segmentation Mask) : 이미지 분할에서의 레이블
- 모델 출력으로 픽셀별 타깃 마스크 생성 → 정보의 공간상 위치에 많은 관심을 둠

# 3. 최신 ConvNet 아키텍처 패턴

- 아키텍처(Architecture) : 모델을 만드는 데 사용된 일련의 선택
- 아키텍처 → 모델 가설 공간(Hypothesis Space) 정의
- 좋은 가설 공간 → 현재 문제와 솔루션에 대한 사전 지식(Prior Knowledge) 인코딩

## 1. 모듈화, 계층화, 재사용

- MHR(Modualrity-Hierachy-Reuse) : 모듈화 + 계층화 + 재사용
- VGG16 아키텍처 : 층 블록 반복, 피라미드 구조 특성 맵
- 그레이디언트 소실 → 층을 쌓을 수 있는 정도에 한계 존재

## 2. 잔차 연결(Residual Connection)

- 그레이디언트 소실 : 너무 깊은 함수 연결 → 잡음이 정보를 압도함, 역전파 미동작
- 층이나 블록의 입력을 출력에 더하는 것
- 그레이디언트 소실 없이 원하는 깊이의 네트워크 생성 가능

## 3. 배치 정규화(Batch Normalization)

- 머신러닝 모델에 주입되는 샘플들을 균일하게 만드는 광범위한 방법
- 데이터 → 정규 분포를 따른다고 가정, 분포를 원점에 맞추고 분산이 1이 되도록 조정
- 훈련하는 동안 현재 배치 데이터의 평균과 분산을 사용하여 샘플 정규화
- 활성화 층 이전에 배치 정규화 층 놓음 → 그레이디언트 전파 도와줌

## 4. 깊이별 분리 합성곱(Depthwise Separable Convolution)

- 입력 채널별로 따로따로 공간 방향의 합성곱 수행
- 점별 합성곱을 통해 출력 채널 합침
- 중간 활성화에 있는 공간상의 위치가 높은 상관관계 가짐
- 채널 간 매우 독립적
- 훨씬 적은 개수의 파라미터, 더 적은 수의 연산 → 일반 합성곱과 유사한 표현 능력

## 5. Xception 유사 모델에 모두 적용

- 모델 → 반복되는 층 블록으로 조직, 일반적으로 여러 개의 합성곱 층 + 최대 풀링 층
- 특성 맵의 공간 방향 크기 감소 → 층 필터 개수 증가
- 깊고 좁은 아키텍처 > 넓고 얕은 아키텍처
- 층 블록에 잔차 연결 → 깊은 네트워크 훈련 가능
- 합성곱 층 다음에 배치 정규화 층 추가
- Conv2D 층 → 파라미터 효율성이 높은 SeparableConv2D 층

# 4. ConvNet이 학습한 것 해석하기

- 해석 가능성(Interpretability) → 컴퓨터 비전 애플리케이션 구축 시 근본적인 문제

## 1. 중간 활성화 시각화

- 어떤 입력이 주어졌을 때 모델에 있는 여러 합성곱과 풀링 층이 반환하는 값을 그리는 것
- 네트워크에 의해 학습된 필터들이 어떻게 입력을 분해하는지 보여줌

## 2. ConvNet 필터 시각화

- 빈 입력 이미지에서 시작해서 특정 필터의 응답을 최대화하기 위해 ConvNet 입력 이미젱 경사 하강법 적용
- 입력 이미지 → 선택된 필터가 최대로 응답하는 이미지

## 3. 클래스 활성화의 히트맵 시각화

- 이미지의 어느 부분이 ConvNet의 최종 분류 결정에 기여하는지 이해하는 데 유용
- ConvNet의 결정 과정 디버깅에 도움
- 이미지에 특정 물체가 있는 위치 파악에 사용
- 클래스 활성화 맵(CAM, Class Activation Map) : 입력 이미지에 대한 클래스 할성화의 히트맵 생성