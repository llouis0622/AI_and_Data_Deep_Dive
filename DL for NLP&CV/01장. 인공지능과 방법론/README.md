# 1. 인공지능이란?

- 패턴 인식, 학습, 의사결정 등 일반적으로 인간 지능이 필요한 작업을 컴퓨터 시스템이 수행하는 기술
- 강인공지능 : 스스로 학습과 인식 등이 가능하며, 지능 또는 지성의 수준이 인간과 근사한 수준까지 이른 경우
- 약인공지능 : 인간이 해결할 수 있으나, 기존의 컴퓨터로 처리하기 힘든 작업을 처리하기 위한 일련의 알고리즘
    - 규칙 기반 AI : 미리 결정된 일련의 규칙 또는 알고리즘에 따라 문제를 해결하거나 작업을 수행함
    - 머신러닝 : 데이터를 이용해 모델을 학습하고 이를 통해 예측이나 분류 수행
    - 딥러닝 : 정보를 처리하고 전송하는 방식을 시뮬레이션하도록 설계된 알고리즘인 인공 신경망 사용

## 1. 인공지능 역사

- 1943년 : 인간 두뇌의 뉴런 작용이 0과 1의 정보 전달로 이뤄지는 개념적 논리 회로 모델 제안
- 1950년 : 엘런 튜링 - 기계의 지능을 판단하는 튜링 테스트 제안
- 1956년 : 존 매카시 - 인공지능 용어 처음 사용
- 1958년 : 프랑크 로잔블라트 - 인간의 신경 세포를 모방한 인공 신경인 퍼셉트론 제시
- 1969년 : 마빈 민스키, 시모어 페퍼트 - 퍼셉트론의 선형 분리 방식 한계 증명
- 1974년 : 퍼셉트론이 복잡한 문제를 해결하지 못해 인공지능이 비판의 대상이 됨
- 1986년 : 제프리 힌턴 - 다층 퍼셉트론과 역전파 알고리즘 증명
- 1991년 : 기울기 소실과 과대적합 문제로 두 번째 암흑기 등장
- 1994년 : 최초의 웹 검색 엔진인 웹크롤러 등장
- 1997년 : IBM - 체스 인공지능 프로그램 딥 블루 승리
- 2006년 : 제프리 힌턴 - 제한된 볼츠만 머신으로 신경망의 학습을 돕는 방법 제안
- 2012년 : 구글, 앤드루 응 - 심층 신경망 구현, 딥러닝 기법 주목
- 2016년 : 딥마인드 - 알파고 승리
- 2017년 : 구글 - Attention Is All You Need 논문에서 트랜스포머 모델 등장
- 2021년 : OpenAI - 텍스트나 이미지 입력 시 그림 생성 DALL.E 출시
- 2023년 : OpenAI - 튜링 테스트를 통과한 GPT-4 출시

## 2. 인공지능 활용 분야

- 의료
- 교통
- 금융
- 제조
- 교육
- 농업 및 에너지
- 전자상거래

# 2. 머신러닝 시스템

- 머신러닝 : 인공지능에 포함되는 영역 중 하나
- 데이터 기반으로 컴퓨터를 프로그래밍하는 연구 분야
- 데이터를 기반으로 학습해 문제 해결 및 시스템 성능 개선
- 머신러닝 → 인공지능에 포함
- 신경망 → 딥러닝 → 머신러닝에 포함
- 인공 신경망 : 인간의 뇌에 있는 신경 세포의 네트워크에서 영감을 얻은 통계학적 학습 알고리즘
    - 서로 연결된 노드의 집합으로 구성, 여러 계층으로 이루어져 있음
- 딥러닝 : 입력층, 은닉층, 출력층으로 이루어짐

## 1. 지도 학습

- 훈련 데이터와 레이블의 관계를 알고리즘으로 학습시키는 방법
- 훈련 데이터 : 입력 데이터 + 출력 데이터

### 1. 회귀 분석

- 둘 이상의 변수 간의 관계 파악 → 독립 변수인 X로부터 연속형 종속 변수인 Y에 대한 모형의 적합도 측정
- 선형 회귀 : 함수가 직선의 특징을 갖고 있음, 중첩의 원리 적용
    - 로버스트 회귀, 라쏘 회귀
    - 단변량 : 종속 변수가 하나일 때, 독립 변수가 하나 이상의 값 사용 가능
    - 다변량 : 종속 변수가 두 개 이상일 때
    - 단순 선형 회귀 분석 : 하나의 종속 변수와 하나의 독립 변수 사이의 관계 분석
    - 다중 선형 회귀 분석 : 하나의 종속 변수와 여러 개의 독립 변수 사이의 관계 분석
- 비선형 회귀 : 방정식이 한 가지 형태로 제한되지 않고 여러 가지 형태인 곡선으로 도출
    - 다층 퍼셉트론 활용

### 2. 분류

- 훈련 데이터에 지정된 레이블과의 관계를 분석해 새로운 데이터의 레이블을 스스로 판별하는 방법
- 새로운 데이터를 대상으로 할당돼야 하는 카테고리, 범주 스스로 판단
- 이진 분류 : 새로운 데이터를 대상으로 참인지 거짓인지 분류 가능
    - 로지스틱 회귀 : 이산형 종속변수를 예측하는 데 로짓 변환 사용
- 다중 분류 : 세 개 이상의 카테고리로 나눠 분류할 수 있을 경우
    - 소프트맥스 회귀 : 가중치를 정규화해 나온 결과값을 모두 더할 때 1이 되게끔 구성하는 것

## 2. 비지도 학습

- 훈련 데이터에 레이블을 포함시키지 않고 알고리즘이 스스로 독립 변수 간의 관계를 학습하는 방법
- 데이터로만 결과 유추

### 1. 군집화

- 입력 데이터를 기준으로 비슷한 데이터끼리 몇 개의 군집으로 나누는 알고리즘
- 같은 군집으로 분류된 데이터 → 서로 비슷한 성질을 가짐
- K-평균 군집화