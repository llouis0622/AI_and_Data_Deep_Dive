## 1. 컴퓨터 비전이란?
- 시각적 데이터들을 효과적으로 이해하고 분석하여 컴퓨터로 하여금 인간의 시각적인 인식능력을 가질 수 있도록 알고리즘을 개발하기 시작한 것

## 2. 비전의 시작
- 약 5억 4천만 년 전, 생물의 증폭 발생
- 눈 -> 능동적인 먹이 사냥 및 보호 활동 가능
- 시각의 중요성

## 3. 컴퓨터 비전의 역사

### 1950년대
- Hubel & Wiesel
- 생물의 시각적 매커니즘 -> 고양이 뇌에 전극을 꽂아 실험
- 시각적 input의  edges가 움직일 때 반응하는 단순한 세포에 초점
- 시각 처리는 edges와 같은 단순한 구조로 시작되어 점점 복잡한 요소들을 처리하고 궁극적으로 실제 시각적 input 인지 가능

### 1960년 초
- Larry Roberts - Block World
- 사물들을 기하학적 모양으로 단순화 -> 시각적 세상 재구성

### 1966년
- MIT Summer Project -> 대부분의 시각 체계 구현

### 1970년 대
- David Marr - VISION, 컴퓨터 비전의 기본서
- Stanford - SRI에서 단순한 모양이나 기하학적 구성을 통해 복잡한 객체를 단순화

### 1980년 대
- David Lowe - 어떻게 하면 단순한 구조로 재구성 할 수 있을까
- line, edge, straight line 조합
- 물체 분할에 초점 -> 이미지의 각 픽셀을 의미 있는 영역으로 분리

### 1999/2000년 대
- 기계학습
- SVM, Boosting, Graphical Models, 초기NN
- Paul Viola, Michael Jones : 실시간 얼굴인식 성공, Adaboost
- David Lowe : SIFT Feature
- 특징기반 객체인식 알고리즘
- 변화에 좀 더 강하고 불변한 특징 발견
- 이미지 전체 매핑 -> 중요한 특징으로 다른 객체에 매핑
- Spatial Pyramid Matching, Support Vector Algorithm
- 21C, 인터넷과 카메라 발전 -> 실험데이터 질 향상
- PASCAL Visual Object Challenge : 알고리즘 테스트 사용, 객체인식 성능 증가
- ImageNet : 가장 큰 데이터셋 제작 -> Overfitting 방지, 일반화 능력 증가 -> 모든 객체 인식
- ILSVRC : 해당 데이터셋으로 지속적인 알고리즘 테스트 진행 -> 오류율 급격히 감소
- 2012년 CNN, 이전 90년 대 LeNet 아키텍쳐에서 계승 but 데이터셋의 연산량 증가 -> 오류율 감소

## 4. cs231n Agenda
- Image Classification
- Object Detection
- Action Classification
- Image Captioning
- 컴퓨터 비전의 최종 목표 : 사람처럼 볼 수 있는 기계 제작