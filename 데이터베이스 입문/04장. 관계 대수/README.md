# 04장. 관계 대수

# 1. 관계 대수 연산

- 관계 대수(Relational Algebra) : 릴레이션을 내부적으로 처리하기 위한 연산들의 집합
- 한 개 이상의 입력 릴레이션으로부터 하나의 새로운 결과 릴레이션 생성
- 집합 연산 그룹 + 관계 연산 그룹

## 1. 집합 연산(Set Operation)

- 릴레이션을 튜플 집합 또는 속성 집합으로 간주하여 이를 처리하는 연산
- 일반적인 수학의 집합 연산과 의미와 기능 동일
- 집합 연산 적용 규칙
    - 집합 연산자는 이항 연산자(Binary Operator) → 피연산자로 두 개의 입력 릴레이션 필요
    - 합집합, 교집합, 차집합 연산 → 두 입력 릴레이션이 서로 합병 가능해야 함
    - 카티션 프로덕트 연산 → 합병 가능 여부와 상관없이 가능

### 1. 합집합(Union, ⋃)

- 두 개의 릴레이션을 합병하여 하나의 릴레이션 반환
- 합집합 후 중복 튜플 제거
- 교환적(Commutative) 특성, 결합적(Associative) 특성
- 차수 : R1 또는 R2 차수와 같음
- 카디널리티 : R1과 R2 카디널리티를 더한 값과 같거나 작음

### 2. 교집합(Intersect, ⋂)

- 두 개의 릴레이션에서 동시에 속하는 공통 튜플만 반환
- 교환적(Commutative) 특성, 결합적(Associative) 특성
- 차수 : R1 또는 R2 차수와 같음
- 카디널리티 : R1 혹은 R2 카디널리티 값보다 작거나 같음

### 3. 차집합(Difference, −)

- 두 개의 릴레이션에서 첫 번째 릴레이션에는 속하지만 두 번째 릴레이션에는 속하지 않는 튜플 반환
- 차수 : R1 또는 R2 차수와 같음
- 카디널리티 : R1 혹은 R2 카디널리티 값보다 작거나 같음

### 4. 카티션 프로덕트(Cartesian Product, ⨉)

- 두 릴레이션의 모든 튜플을 수평으로 결합
- 첫 번째 릴레이션의 각 튜플에 대해 두 번째 릴레이션의 모든 튜플을 반복하여 앞뒤로 연결
- 특정 릴레이션의 튜플을 다른 릴레이션의 튜플과 연관시키기 위하여 결합할 때 사용
- 교환적(Commutative) 특성, 결합적(Associative) 특성
- 차수 : R1과 R2 차수를 더한 값과 같음
- 카디널리티 : R1과 R2 카디널리티를 곱한 값과 같음

## 2. 관계 연산(Relation Operation)

- 릴레이션의 구조적 특성에 기반을 둔 연산 포함

### 1. 셀렉트(Select, σ)

- 릴레이션에서 특정 튜플 추출
- 단항 연산
- 릴레이션의 튜플 중에서 명세된 선택_조건식을 만족하는 튜플로만 구성 → 릴레이션 수평 분할

    ```sql
    𝜎선택_조건식(R)
        선택_조건식 = {단순_조건식|복합_조건식}
        단순_조건식 = ⟨R의 속성이름⟩ ⟨비교연산자⟩ ⟨상수_값(또는 R의 속성이름)⟩
        복합_조건식 = 단순_조건식[⟨논리연산자⟩ [단순_조건식]]*
    ```


### 2. 프로젝트(Project, 𝛱)

- 릴레이션에서 특정 속성 추출
- 단항 연산
- 릴레이션에 속한 속성 중에서 속성_리스트에 명세된 속성만 구성 → 릴레이션 수직 분할

    ```sql
    𝛱속성_리스트(R)
        속성_리스트 = 속성명1, 속성명2, ...
    ```


### 3. 조인(Join, ⋈)

- 두 릴레이션의 공통 속성을 기준으로 조인 조건을 만족하는 튜플을 수평으로 결합
- 첫 번째 릴레이션의 각 튜플에 대해 두 번째 릴레이션의 모든 튜플을 앞뒤로 반복하여 연결한 튜플 조합 중 조인_조건식을 만족하는 튜플만 구성

    ```sql
    R1⋈조인_조건식R2
        조인_조건식 = 참, 거짓을 판별할 수 있는 논리식
    ```


1. 세타 조인(Theta Join)
- 조인_조건식에 6개의 𝛉 비교_연산자 중 하나 사용

    ```sql
    R1⋈a1=a2R2
    R1⋈a1!=a2R2
    R1⋈a1<a2R2
    R1⋈a1>a2R2
    R1⋈a1<=a2R2
    R1⋈a1>=a2R2
    ```


1. 동등 조인(EquiJoin)
- 조인_조건식에 특별히 = 비교 연산자를 사용하는 세타 조인

    ```sql
    R1⋈조인_조건식R2 = R1⋈a1=a2R2
    ```


1. 자연 조인(Natural Join)
- 동등 조인 결과 중에서 조인_조건식에 사용된 중복 속성 자동 제거

    ```sql
    R1⋈N(조인_속성_리스트)R2
    ```


### 4. 디비전(Division, ÷)

- 특정 값들을 모두 가지고 있는 튜플 찾기
- R2의 모든 튜플에 연관된 R1의 튜플 중에서 R2에 속한 속성을 제외한 나머지 속성 값만으로 구성된 릴레이션

### 5. 기본 연산

- 기본 연산(Primitive Operation) : 어떤 연산을 조합하더라도 같은 결과를 반환받을 수 없음
    - 합집합, 차집합, 카티션 프로덕트, 셀렉트, 프로젝트
- 복합 연산(Composite Operation) : 여러 연산들을 조합하면 같은 연산 결과 반환 가능
    - 교집합, 조인, 디비전

## 3. 확장 연산

- 기존 관계 대수 연산을 확장하여 추가로 정의

### 1. 세미 조인(SemiJoin, ⋉, ⋊)

- 자연 조인이 반환하는 결과 릴레이션 중에서 한쪽 릴레이션 속성만으로 한정하여 반환
- 왼쪽 세미 조인(Left SemiJoin) : 자연 조인 결과 중 왼쪽 릴레이션의 속성 반환
- 오른쪽 세미 조인(Right SemiJoin) : 자연 조인 결과 중 오른쪽 릴레이션의 속성 반환

### 2. 외부 조인(Outer Join, ⟗, ⟕, ⟖)

- 자연 조인 결과에 포함되지 않는, 조인에 실패한 튜플까지 모두 포함하도록 확장한 연산
- 완전 외부 조인(Full Outer Join) : 왼쪽과 오른쪽 R1, R2 릴레이션의 모든 튜플을 빠짐없이 조인 결과에 포함하도록 대응 튜플이 없는 경우 널 값을 채워서 반환
- 왼쪽 외부 조인(Left Outer Join) : 왼쪽 R1 릴레이션의 모든 튜플을 빠짐없이 조인 결과에 포함하도록 대응 튜플이 없는 경우 널 값을 채워서 반환
- 오른쪽 외부 조인(Right Outer Join) : 오른쪽 R2 릴레이션의 모든 튜플을 빠짐없이 조인 결과에 포함하도록 대응 튜플이 없는 경우 널 값을 채워서 반환

### 3. 외부 합집합(Outer Union, ⋃+)

- 부분적으로만 합병 가능한 두 릴레이션의 튜플 합병

# 2. 관계 대수의 활용

## 1. 질의 요청의 관계 대수식 표현

- 질의문의 관계 대수식 작성
- 복수의 관계 대수식 작성

## 2. 질의 트리 작성 및 최적화

- 질의 트리(Query Tree) : 트리 형태의 계층적 구조로 관계 대수식을 표현한 것
- 관계 대수식에서 가장 먼저 실행해야 할 연산부터 질의 트리의 가장 아래쪽에 표현
- 질의 실행 계획(Query Execution Plan) : 여러 가능한 후보 질의 트리들을 최적에 가까운 트리로 변환
- 질의 최적화(Query Optimization) : 연산 순서를 조정함으로써 연산으로 생기는 중간 릴레이션 크기 최소화
- 질의 트리 최적화 변환 규칙
    - AND 연산자로 연결된 셀렉트 연산 → 분리하여 개별 셀렉트 연산 변환
    - 셀렉트 연산 → 가능한 먼저 실행되도록 질의 트리 아래쪽으로 이동
    - 프로젝트 연산 → 프로젝트 속성 분리 후 개별 프로젝트 연산 변환 → 질의 트리 아래쪽으로 이동
    - 여러 셀렉트 연산 중 결과 릴레이션 크기가 가장 작은 것부터 제한적 셀렉트 연산 순 배열
    - 카티션 프로덕트 연산 + 셀렉트 연산 → 조인 연산으로 통합 변환
    - OR 연산 → 가능한 AND 연산으로 변환