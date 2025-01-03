# 10장. 데이터베이스 설계

# 1. 데이터베이스 설계

## 1. 데이터 모델링과 데이터 모델

### 1. 데이터 모델링

- 데이터베이스 모델링(Data Modeling) : 데이터베이스 구조 생성 과정
- 데이터 모델(Data Model) : 각 모델링 단계마다 데이터베이스 구조를 명세하기 위한 다양한 표현 도구 사용

### 2. 데이터 모델링과 데이터 모델

- 개념적 데이터 모델링(Conceptual Data Modeling) : 분석 관점에서 사용자 요구 사항을 충실히 표현, E-R 모델
- 논리적 데이터 모델링(Logical Data Modeling) : 설계 관점에서 개념적 모델을 충실히 변환하여 표현, 특정 유형군의 DBMS, 관계형 데이터 모델
- 물리적 데이터 모델링(Physical Data Modeling) : 선택한 특정 DBMS의 특성과 구조에 적합한 물리적 데이터 구조 명세

## 2. 데이터베이스 설계 과정

- 1단계 - 요구사항 분석 : 요구사항 명세서(Requirement Specification) 작성
- 2단계 - 개념적 설계 : E-R 다이어그램 작성
- 3단계 - 논리적 설계 : 논리적 데이터베이스 스키마 완성
- 4단계 - 물리적 설계 : 물리적 데이터베이스 스키마 완성
- 5단계 - 구현

# 2. 요구사항 분석

## 1. 요구사항 명세

- 요구사항 분석 단계 : 구축하고자 하는 데이터베이스의 구현 범위와 사용자의 범주 결정
- 분석 결과 → 요구사항 명세서로 문서화

## 2. 요구사항 명세서 작성

- 사용자의 요구 사항만을 충실히 반영
- 구체적 문장 형태로 서술

# 3. 개념적 설계

- 데이터베이스 설계의 전체 골격을 결정하는 과정
- 요구사항 명세서 → 핵심 데이터 요소 추출 → E-R 다이어그램 작성

## 1. 개체 정의

- 요구사항 명세서 문장 중 주어나 목적어로 표현되는 명사 찾기
- 속성에 해당하는 하위 개념의 명사 → 꾸밈을 받는 상위 개념의 명사
- 실세계에서 독립적 존재 → 꾸밈을 받는 하위 개념 명사 중에 고유한 명칭 가짐
- 상대적으로 오랜 시간 지속
- 또 다른 여러 개체들이 공유하는 대상

## 2. 관계 정의

- 요구사항 명세서 문장 중 서술어로 표현되는 동사
- 반드시 연관성을 갖는 둘 이상의 개체 필요
- 의미를 명확히 표현하거나 꾸며주는 하위 개념의 속성
- 종속적 존재 → 고유한 명칭 미보유
- 일시적, 짧은 시간만 지속

## 3. 속성 정의

- 주어, 목적어, 서술어를 수식하거나 꾸며주는 하위 명사
- 공통의 특정 대상을 명확히 표현하기 위해 연관성을 가짐
- 상위 개념의 명사, 하위 개념의 명사
- 종속적 → 상위 개념 반드시 필요

## 4. E-R 다이어그램의 작성

### 1. 유의 사항

- 요구사항 명세서 문장 안 개체 → 사각형, 관계 → 마름모, 속성 → 타원
- 의미가 중복되거나 반복 → 하나의 표준 용어로 통일
- 데이터와 직접적인 연관성이 낮거나 너무 일반적인 표현 삭제
- 데이터 관점에서 불필요하거나 반복되는 기능 무시
    - 정보, 데이터 : 모든 특성에 관련되는 공통 표현
    - 유지, 저장, 관리, 기록, 보관 : 데이터베이스의 일반 기능
    - 입력, 수정, 삭제, 검색 : 데이터베이스 시스템의 일반 기능
    - 발급, 부여, 선택, 확인, 취소 : 관리자나 사용자의 소프트웨어 기능

# 4. 논리적 설계

## 1. 개체 변환

- 개체 릴레이션(Entity Relation) : 새로 생성된 릴레이션을 E-R 다이어그램의 개체 표현
- 개체의 키 속성 → 릴레이션의 기본키 속성, 일반 속성 → 릴레이션의 속성, 개체 속성 이름 → 릴레이션 속성 이름

## 2. 관계 변환

### 1. 일대다(1:N) 관계 변환

- 하나의 외래키 속성으로 변환
- 1측 개체 릴레이션의 기본키 속성을 N측 개체 릴레이션에 외래키 속성으로 추가
- 일대다 관계의 모든 하위 속성 → 외래키를 추가한 N측 개체 릴레이션의 속성으로 변환

### 2. 일대일(1:1) 관계 변환

- 하나의 외래키 속성으로 변환
- 한쪽 개체 릴레이션의 기본키 속성을 다른쪽 개체 릴레이션에 외래키 속성으로 포함
- 일대일 관계의 모든 속성 → 외래키를 추가한 릴레이션 속성으로 변환

### 3. 다대다(M:N) 변환

- 하나의 독립된 릴레이션으로 변환, 관계 이름 → 새로운 릴레이션 이름
- 관계 속성 → 릴레이션 속성, 관계 속성 이름 → 릴레이션 속성 이름
- 관계 릴레이션