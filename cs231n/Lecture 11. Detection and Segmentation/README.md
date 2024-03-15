## 1. Image Segmentation
- Medical Imagine
- Semantic Segmentation
- Classification + Localization
- Object Detection
- Instance Segmentation

## 2. Semantic Segmentation
- Image 단위로 Labeling X, 모든 픽셀에 대해 Independently Labeling -> 픽셀 단위로 Classification
- crop -> 옆으로 계속 슬라이싱하면서 Center Pixel 예측 가능
- Conv -> High Resolution Image 프로세싱 가능

## 3. Fully Convolutional
- Dowmsampling, Upsampling -> High Resolution Image로 복구
- FC를 통해 Transitioning 하는 것과 달리 Spatial Resolution 계속 증가
- input image size = output image size
- lower spatial resolution에서 processing -> Comutationally Efficient

## 4. Max Unpooling
- 같은 Position에 Max Value가 오도록 Constraint를 걸어 Image의 사이즈 증가

## 5. Transpose Convolution
- stride -> down sampling
- 1개의 픽셀에 vector 곱함 -> stride 주고 image size 증가
- input -> output mapping, size 증가

## 6. Convolution as Matrix Multiplication
- transpose convolution과 일반 convolution 비슷한 현상 발생
- input image tensor를 kernel 곱함 -> 학습 가능

## 7. Classification + Localization
- multitask
- Loss를 hyperparameter를 끼고 weighted sum
- 2개의 테스크 -> Fine Tuning

## 8. Object Detection
- 여러 개의 object에 대해 annotation
- 매번 다른 crop -> 슬라이딩
- object가 있는 곳을 먼저 proposal 알고리즘 사용
- Region Proposals -> Fixed Way

## 9. R-CNN
- Fixed Way Regin Proposal Network 사용
- 시간 증가 -> Candidate 증가 -> 학습량 증가
- Fast R-CNN
  - Conv -> High Resolution Feature Map -> Fixed Way Selective Search
  - CNN으로 Feature Map을 딸 때 Proposal 같이 학습