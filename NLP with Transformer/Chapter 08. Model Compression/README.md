# 1. 의도 탐지 예제

- CLINC150 데이터셋 - BERT 베이스 모델 사용

# 2. 벤치마크 클래스 만들기

- 여러 가지 제약 조건
    - 모델 성능 : 오류 발생 시 손실 비용이 크거나 수백만 개의 샘플에서 추론해야 함
    - 레이턴시 : 대량의 트래픽을 처리하는 실시간 환경
    - 메모리 : 파라미터의 크기에 따라 기가바이트 단위의 디스크 & 램 필요

# 3. 지식 정제로 모델 크기 줄이기

- 작은 스튜던트 모델을 훈련하는 방법
- 느리고 크지만 성능이 더 높은 티처의 동작을 모방하도록 모델 훈련

## 1. 미세 튜닝에서의 지식 정제

- 티처의 소프트 확률로 정답 레이블 보강 → 모델 학습 시 부가 정보 제공

## 2. 사전 훈련에서의 지식 정제

- 사전 훈련 시 후속 작업에서 미세 튜닝이 가능한 범용 스튜던트 생성
- 미세 튜닝된 모델 → 더 작고 빠른 모델로 미세 튜닝

## 3. 지식 정제 트레이너 만들기

- 새로운 하이퍼파라미터 추가 → 정제 손실의 상대적인 가중치 제어, 레이블의 확률 분포 완만함 조절
- 미세 튜닝한 티처 모델
- 크로스 엔트로피, 지식 정제 손실을 연결한 새로운 손실 함수

## 4. 좋은 스튜던트 선택하기

- 사전 훈련된 언어 모델 중 선택
- 레이턴시, 메모리 사용량 감소 모델 선택
- 쿼리 토큰화 & 인코딩

## 5. 옵투나로 좋은 하이퍼파라미터 찾기

- 옵투나(Optuna) : 최적화 프레임워크, 검색 문제를 여러 시도를 통해 최적화할 목적 함수로 표현
- 여러 시도 → 하나의 스터디로 수집

## 6. 정제 모델 벤치마크 수행하기

- 파이프라인 구성 → 벤치마크 재수행

# 4. 양자화로 모델 속도 높이기

- 양자화(Quantization) : 부동 소수점 숫자를 이산화할 수 있음에서 시작
- 계산량 감소 → 가중치와 활성화 출력을 정밀도가 낮은 데이터 타입으로 표현
- 동적 양자화 : 추론 과정에만 적응, 모델 가중치 변환
- 정적 양자화 : 양자화 체계를 사전에 계산해 부동 소수점 변환을 피함
- 양자화를 고려한 훈련 : 가짜로 양자화 효과 흉내내기 가능

# 5. 양자화된 모델의 벤치마크 수행하기

- 양자화된 모델 → 성능 증가

# 6. ONNX와 ONNX 런타임으로 추론 최적화하기

- ONNX : 파이토치, 텐서플로 등 다양한 프레임워크에서 딥러닝 모델을 나타내기 위해 공통 연산자와 공통 파일 포맷을 정의하는 공개 표준
- 연산자 융합, 상수 폴딩 등 그래프 최적화 도구 제공
- ONNX 포맷 변환
    - 하나의 파이프라인으로 모델 초기화
    - 계산 그래프 기록 → 플레이스홀더 입력 → 파이프라인 실행
    - 동적 시퀀스 길이 처리 → 동적인 축 정의
    - 네트워크 파라미터와 함께 그래프 저장

# 7. 가중치 가지치기로 희소한 모델 만들기

## 1. 심층 신경망의 희소성

- 가중치 연결 → 점진적 제거 → 모델 희소
- 파라미터 개수 감소, 양자화

## 2. 가중치 가지치기 방법

- 중요도 점수(Importance Score) : 행렬 계산 후 중요도 순으로 가중치 선택
- 가지치기 방법
    - 어떤 가중치를 삭제해야 하는가
    - 최상의 성능을 내려면 남은 가중치를 어떻게 조정해야 하는가
    - 이런 가지치기 계산을 효율적으로 수행하는 방법은 무엇인가
- 절댓값 가지치기 : 가중치 절댓값 크기에 따라 점수 계산
- 이동 가지치기 : 미세 튜닝 중 점진적 가중치 제거를 통해 희소 모델 생성