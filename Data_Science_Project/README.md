# 1. 분류모델 기초

---

# 1. 사이킷런과 머신러닝

## 1. 사이킷런 소개

- 파이썬 머신러닝 라이브러리
- Classification(분류), Regression(회귀) 모델 주로 사용
- Supervised Learning(지도 학습) : 정답이 있는 문제
- Unsupervised Learing(비지도 학습) : 정답이 없는 문제
    - Clustering(클러스터링)
    - Dimensionality Reduction(차원 축소)

## 2. 사이킷런 활용 흐름

- Training : Model ← Training Data + Training Labels
- Generalization
    - Prediction ← Test Data
    - Evaluation ← Test Labels
- Linear Models
    - LogisticRegressionCV
    - RidgeCV
    - RidgeClassifierCV
    - LarsCV
    - ElasticNetCV
- Feature Selection
    - RFECV
- Tree-Based Models
    - DecisionTreeCV
    - RandomForestClassifierCV
    - GradientBoostingClassifierCV

## 3. 사이킷런의 의사결정나무 알고리즘 알아보기

- Supervised Learning - Classification 기법
- Desicion Tree 알고리즘 : 분류 및 회귀 모델 적용 가능

# 2. 의사결정나무로 간단한 분류 예측 모델 만들기

## 1. 당뇨병 데이터셋 소개

- Pregnancies : 임신 횟수
- Clucose : 2시간 동안의 경구 포도당 내성 검사에서 혈장 포도당 농도
- BloodPressure : 이완기 혈압(mm Hg)
- SkinThickness : 삼두근 피부 주름 두께(mm), 체지방을 추정하는 데 사용되는 값
- Insulin : 2시간 혈청 인슐린(mu U / ml)
- BMI : 체질량 지수(체중kg / 키(m)^2)
- DiabetesPedigreeFuncion : 당뇨병 혈통 기능
- Age : 나이
- Outcome : 768개 중에 268개의 결과 클래스 변수(0, 1)는 1이고 나머지는 0

## 2. 학습과 예측을 위한 데이터셋 만들기

- 필요한 라이브러리 로드
- 데이터셋 로드
- 학습, 예측 데이터셋 나누기
- 학습, 예측에 사용할 컬럼
- 정답값이자 예측해야 될 값(컬럼)
- 학습, 예측 데이터셋 만들기

## 3. 의사결정나무로 학습과 예측하기

- 머신러닝 알고리즘 가져오기
- 학습(훈련)
- 예측

## 4. 예측한 모델의 성능 측정하기

- 트리 알고리즘 분석하기
- 정확도 예측하기

---

# 2. EDA를 통해 데이터 탐색하기

# 1. 당뇨병 데이터셋 미리보기

- Pregnancies : 임신 횟수
- Glucose : 2시간 동안의 경구 포도당 내성 검사에서 혈장 포도당 농도
- BloodPressure : 이완기 혈압 (mm Hg)
- SkinThickness : 삼두근 피부 주름 두께 (mm), 체지방을 추정하는데 사용되는 값
- Insulin : 2시간 혈청 인슐린 (mu U / ml)
- BMI : 체질량 지수 (체중kg / 키(m)^2)
- DiabetesPedigreeFunction : 당뇨병 혈통 기능
- Age : 나이
- Outcome : 768개 중에 268개의 결과 클래스 변수(0 또는 1)는 1이고 나머지는 0입니다.

# 2. 결측치 보기

- 결측치 처리 및 시각화

# 3. 훈련과 예측에 사용할 정답값을 시각화로 보기

- 정답값
- countplot

# 4. 두 개의 변수를 정답값에 따라 시각화 해보기

- barplot
- boxplot
- violinplot
- swarmplot

# 5. 수치형 변수의 분포를 정답값에 따라 시각화 해보기

- distplot

# 6. 서브플롯으로 모든 변수 한 번에 시각화 하기

- subplot
- distplot

# 7. 시각화를 통한 변수간의 차이 이해하기

- regplot
- lmplot
- pairplot
- PairGrid

# 8. 피처엔지니어링을 위한 상관 계수 분석하기

- corr()
- heatmap()
- lmplot()

---

# 3. 탐색한 데이터로 모델성능 개선

# 1. 연속 수치 데이터를 범주형 변수로 변경하기

- 오버피팅(Overfitting)
- 언더피팅(Underfitting)
- Pregnancies : 임신 횟수
- Glucose : 2시간 동안의 경구 포도당 내성 검사에서 혈장 포도당 농도
- BloodPressure : 이완기 혈압 (mm Hg)
- SkinThickness : 삼두근 피부 주름 두께 (mm), 체지방을 추정하는데 사용되는 값
- Insulin : 2시간 혈청 인슐린 (mu U / ml)
- BMI : 체질량 지수 (체중kg / 키(m)^2)
- DiabetesPedigreeFunction : 당뇨병 혈통 기능
- Age : 나이
- Outcome : 768개 중에 268개의 결과 클래스 변수(0 또는 1)는 1이고 나머지는 0입니다.

# 2. 범주형 변수를 수치형 변수로 변환하기 - 원핫인코딩

- 원핫인코딩 : 해당 컬럼에 해당되는지의 여부 확인
- 범주형 데이터 → 수치형 변환

# 3. 결측치 평균값으로 대체하기

- isnull().sum()

# 4. 결측치 중앙값으로 대체하기

- isnull().sum()
- 평균값을 결측치로 채워넣는 것보다 성능 향상

# 5. 수치형 변수를 정규분포 형태로 만들기

- 왜도 : 치우친 정도
- 첨도 : 뾰족한 정도
- 로그 변환

# 6. 상관 분석을 통해 파생변수 만들기

- crosstab()
- lmplot()

# 7. 이상치 다루기

- boxplot()
- IQR

# 8. 피처 스케일링

- StandardScaler()
- fit(), transform()

# 9. 전처리한 피처를 CSV 파일로 저장하기

- to_csv