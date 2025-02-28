# 1. 선형대수와 머신러닝의 관계

- 선형대수 → 행렬 → 데이터 다루기 유리

# 2. 행렬의 기초

## 1. 행렬이란

- 데이터 테이블 형태로 표현 → 행렬의 형태로 표현 가능
- 행렬 : 행과 열로 구성, 스칼라 혹은 벡터로 구성
    - 행 : 가로 방향
    - 열 : 세로 방향
    - 스칼라 : 행렬을 구성하는 요소, 각 숫자, 행렬의 구성 요소 중 최소 단위
    - 벡터 : 스칼라의 집합, 크가와 방향을 가짐
- 행렬 원소 : 행렬을 구성하는 각 스칼라값, $a_{ij}$
- 텐서 : n차원으로 일반화한 행렬

## 2. 대각 행렬(Diagonal Matrix)

- 대각 행렬 : 행렬의 대각 원소 이외의 모든 성분이 0인 행렬, $D$
- 단위 행렬(Identity Matrix) : 행 번호와 열 번호가 동일한 주 대각선의 원소가 모두 1이며 나머지 원소는 모두 0인 정사각 행렬, $I$
- 정사각 행렬(Square Matrix) : 행과 열의 크기가 같은 행렬

## 3. 전치 행렬(Transposed Matrix)

- 전치 행렬 : 기존 행렬의 행과 열을 바꾸는 행렬, $A^T$
- 행이 열이되고 , 열이 행이 된 것, $a_{ij} \rightarrow a_{ji}$

## 4. 행렬의 덧셈, 뺄셈

- 행렬 간 덧셈, 뺄셈 가능
- 각 행렬에 대응되는 원소를 더하거나 뺌

## 5. 행렬의 스칼라곱

- 모든 행렬 원소에 곱하려는 스칼라를 곱함
- 행렬을 구성하는 원소 배를 함, 길이가 배가 됨

## 6. 행렬곱

- 행렬 간 서로 곱하는 것
- 앞 행렬의 열 크기와 뒷 행렬의 행 크기가 일치해야 계산 가능
- $A(m * r) B(r * n) \rightarrow AB(m * n)$
- 두 행렬을 곱할 때 앞에 위치하는 행렬의 행과 뒤에 위치하는 열의 원소를 각각 곱한 후 더하는 것

## 7. 행렬의 원소곱

- 차원이 동일한 두 행렬의 동일 위치 원소를 서로 곱하는 방법
- $(A ⊙ B)_{ij} = a_{ij}b_{ij}$

## 8. 행렬식(Determinant)

- 행렬의 특성을 하나의 숫자로 표현하는 방법 중 하나
- 행렬식의 절대값 : 해당 행렬이 단위 공간을 얼마나 늘렸는지 혹은 줄였는지를 나타냄

## 9. 역행렬(Inverse Matrix)

- 행렬 A에 대해 AB = I를 만족하는 행렬 B가 존재함
- $AA^{-1} = A^{-1}A = I$
- 가역 행렬 : 해당 행렬의 행렬식이 0이 아니면 역행렬 존재

# 3. 내적(Inner Product)

- 벡터와 벡터의 연산 결과값이 스칼라로 나옴
- 내적을 구하려는 각 벡터의 요소를 서로 곱한 후 더하는 것
- $<u, v> = u \cdot v = u_1v_1 + u_2v_2 + … + u_nv_n$
- $u \cdot v = u^Tv$
- 벡터의 길이, 벡터 사이 관계 파악 가능
- 내적 > 0 → 두 벡터 사이의 각도 < 90
- 내적 < 0 → 두 벡터 사이의 각도 > 90
- 내적 = 0 → 두 벡터 사이의 각도 = 90
- norm : 벡터의 길이, $||u|| = \sqrt{u^2_1 + u^2_2 + ... + u^2_n}$
- $u \cdot v = ||u|| \ ||v|| cos \theta$
- 정사영(Projection) : 한 벡터가 다른 벡터에 수직으로 투영하는 것, $||u|| cos \theta$

# 4. 선형 변환(Linear Transformation)

- 두 벡터 공간 사이의 함수
- 벡터 확대, 축소, 회전, 반사

# 5. 랭크, 차원

## 1. 벡터 공간(Vector Space), 기저(basis)

- 벡터 공간 : 벡터 집합이 존재할 때, 해당 벡터들로 구성할 수 있는 공간
- 기저 : 벡터 공간을 생성하는 선형 독립인 벡터들
- 2차원 평면을 구성하는 기저 : (x, 0), (0, y)

## 2. 랭크와 차원(Dimension)

- 행공간 : 행벡터로 span할 수 있는 공간
- 열공간 : 열벡터로 span할 수 있는 공간
- 차원 : 기저 벡터의 개수, 벡터 공간을 구성하는 데 필요한 초소한의 벡터 개수

## 3. 직교 행렬(Orthogonal Matrix)

- 어떤 행렬의 행벡터와 열벡터가 유클리드 공간의 정규 직교 기저를 이루는 행렬
- 행렬을 구성하는 각 행벡터 혹은 열벡터의 길이가 1이며 서로 수직인 벡터로 이루어진 행렬
- $AA^T = A^TA = I$
- 직교 행렬의 전치 행렬 → 직교 행렬
- 직교 행렬의 역행렬 → 직교 행렬
- 직교 행렬끼리의 곱 결과 → 직교 행렬
- 직교 행렬의 행렬식 → 1, -1

# 6. 고윳값(Eigenvalue), 고유 벡터

- 고윳값 : 특성값, 선형 변환 이후 변한 크기
- 고유 벡터 : 특성 벡터, 벡터에 선형 변환을 취했을 때, 방향은 변하지 않고 크기만 변하는 벡터

# 7. 특이값 분해

## 1. 닮음

- $P^{-1}AP = B$를 만족하는 가역 행렬 P가 존재할 때, 정사각 행렬 A, B는 서로 닮음임

## 2. 직교 대각화(Orthogonal Diagonalization)

- 대각 행렬이 존재하는 경우 → 직교 대각화
- 행렬에 선형 변환을 취한 결과, 대각 원소만 남는 대각 행렬이 되는 것
- 공분산 행렬

## 3. 고윳값 분해(Eigenvalue Decomposition)

- 직교 대각화의 한 종류
- 행렬을 고유 벡터, 고윳값의 곱으로 분해하는 것

## 4. 특이값 분해(Singular Value Decomposition)

- 대상 행렬을 m*n 행렬로 일반화시킨 것
- 차원 축소 : 데이터 전체 공간의 차원보다 낮을 차원으로 적합시킬 수 있는 기존 행렬의 차원보다 낮은 차원의 공간을 찾는 것

# 8. 이차식 표현

## 1. 이차식 개념

- 이차식 표현 : 다항식을 벡터 형태로 나타낼 때 사용하는 유용한 방법
- $w_1x_1 + w_2x_2 + … w_px_p$
- $w_1x_1^2 + w_2x_2^2 + … w_px_p^2 \rightarrow w_1x_1^2 + w_2x_2^2 + … 2w_3x_1x_2$

## 2. 양정치 행렬

- 양정치(Positive Definite) : $x^TWx > 0, \text{for all} \ x ≠ 0$
- 음정치(Negative Definite) : $x^TWx < 0, \text{for all} \ x ≠ 0$
- 양정치 행렬 : 행렬의 고윳값이 모두 0보다 큰 행렬
- 음정치 행렬 : 행렬의 고윳값이 모두 0보다 작은 행렬

# 9. 벡터의 미분

- $\frac{\partial (w^TX^TXw)}{\partial w} = 2X^TXw$