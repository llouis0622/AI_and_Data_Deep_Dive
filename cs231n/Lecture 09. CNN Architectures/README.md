## 1. LeNet-5
- 산업에 성공적으로 적용된 최초의 ConvNet
- 이미지를 입력으로 받아 stride=1인 5*5 필터를 거치고 몇 개의 Conv와 Pooling Layer 거침
- 마지막에 FC

## 2. 2012년 AlexNet
- 최초의 Large Scale CNN
- ImageNet Classification Task 성능 우수
- ConvNet 연구 유행 시작

## 3. ZFNet
- AlexNet의 하이퍼 파라미터 개선
- AlexNet과 레이어 수 동일, 구조 동일

## 4. VGGNet
- 훨씬 깊어짐
- 작은 필터 3 * 3 -> Depth 키우기 가능
- 3 * 3 필터 3개와 7 * 7 필터 1개의 Receptive Field 동일
- 작은 필터 -> Parameter 감소, 비선형성 증가, 깊이 증가

## 5. GoogleNet
- Inception Module을 여러 개 쌓아서 만듦
  - Network 안에 Network 느낌으로 Local Topology 만든 것
- 파라미터를 줄이기 위해 FC 레이어 제거 -> 더 깊게 구현 가능
- 동일한 입력을 받는 여러 개의 필터 병렬 존재 -> 각각의 출력값들을 Depth 방향으로 합침, 이후 하나의 Tensor를 다음 레이어로 전달
- 1 * 1 Conv 필터 -> 입력의 Depth를 줄여 연산량 감소 가능
- 정보 손실 가능성 존재 but Redundancy가 있는 Input Features 선형결합 -> 연산량 감소, 비선형 레이어 추가

## 6. ResNet
- 엄청 깊어짐
- Residual Connections 사용
- 모델이 깊어질수록 최적화가 어려워질 것이라 생각
- 모델이 깊다면 최소한 더 Shallow한 모델의 성능만큼은 나와야 하지 않는가 의문 제기
- Map 전체를 Average Pooling
- Depth >= 50 -> Bottleneck Layer 추가, 3 * 3 Conv 연산량 감소

## 7. 모델별 Complexity
- x축 : 연산량
- y축 : Accuracy
- Google-Inception V4 : 가장 좋은 모델
- VGGNet : 효율성 작음, 메모리 효율 낮음, 계산량 많음
- GoogleNet : 효율적, 메모리 사용 적음
- AlexNet : 초기 모델, 성능 별로
- ResNet : 적당한 효율성, 정확성 최상위

## 8. Network in Network(NiN)
- 각 Conv Layer 안에 Multi Perceptron(FC)를 쌓아서 네트워크 안에 작은 네트워크를 만드는 방식
- 1 * 1 Conv Layer 사용 -> Abstract Features 잘 뽑을 수 있도록 함

## 9. ResNet 블록 향상
- Direct Path 증가 -> Forward, Backprop 잘 될 수 있도록 설계
- Filter의 Width 증가 -> 병렬화 -> 계산 효율 증가
- ResNet + Inception : Residual Block 내에 다중 병렬 경로 추가