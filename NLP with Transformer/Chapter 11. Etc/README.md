# 1. 트랜스포머 확장

## 1. 규모의 확장

- 컴퓨팅 예산, 데이터셋 크기, 모델 크기에 따라 비교 가능
- 성능과 규모의 비례 관계
- 거듭제곱 법칙 : 테스트 손실 → 여러 자릿수에서 거듭제곱 관계 성립
- 샘플 효율성

## 2. 규모 확장의 어려움

- 인프라 : GPU 및 통신 병목 등
- 비용 : 최대 규모 훈련 시 팀 및 자원 감당 불가
- 데이터셋 큐레이션
- 모델 평가
- 배포
- BigScience : 대규모 언어 모델에 초점을 둔 연구 워크숍
- EleutherAI : AI 연대, 확장, 오픈소스 AI 연구에 주로 관심이 있는 자원 봉사 연구원, 엔지니어, 개발자가 만든 분산형 단체

## 3. 어텐션

- Attention Is All You Need

## 4. 희소 어텐션

- 글로벌 어텐션 : 시퀀스에서 다른 모든 토큰에 주의를 기울이는 몇 개의 특수한 토큰 정의
- 밴드 어텐션 : 대각선에 걸친 어텐션 계산
- 팽창 어텐션 : 간격을 둔 팽창 윈도우를 사용해 일부 쿼리-키 쌍을 건너뜀
- 랜덤 어텐션 : 쿼리마다 몇 개의 키를 랜덤하게 샘플링해 어텐션 점수 계산
- 블록 로컬 어텐션 : 시퀀스를 블록으로 나누고 이 블록 안에서 어텐션 점수 계산

## 5. 선형 어텐션

- 어텐션 점수 계산에 관련된 연산 순서 변경
- 유사도 함수를 두 부분으로 나누는 커널 함수로 표현

# 2. 텍스트를 넘어서

- 사람이 만든 편향
- 상식
- 사실
- 데이터 형태

## 1. 비전

- iGPT
- ViT

## 2. 테이블

- TAPAS

# 3. 멀티모달 트랜스포머

- 스피치-투-텍스트
- 비전-텍스트
    - VQA
    - LayoutLM
    - DALL.E
    - CLIP