# 13장. NoSQL과 몽고DB

# 1. NoSQL 몽고DB 개요

- 기존 데이터베이스 기술이 새로운 환경의 데이터와 사용자, 작업의 규모에 효율적이지 못함
- 키-값 DB : 속성 이름에 해당하는 키와 속성 값에 해당하는 키값의 쌍으로 데이터 저장
    - 성능 높음, 확장성 높음, 유연성 높음, 복잡성 낮음
    - 다이나모, 레디스
- 문서 DB : 키에 대응하는 키값을 반구조화된 문서 형식으로 데이터 저장
    - 성능 높음, 확장성 높음, 유연성 높음, 복잡성 낮음
    - 몽고DB, 카우치DB
- 컬럼패밀리 DB : 키에 대응되는 컬럼과 값의 조합을 다양하게 열 단위로 묶어서 테이블 형식으로 데이터 저장
    - 성능 높음, 확장성 높음, 유연성 보통, 복잡성 낮음
    - 카산드라, HBase
- 그래프 DB : 객체 데이터는 노드 안에, 객체 사이의 연관성 데이터는 관계 링크 안에 표현하여 그래프 구조로 데이터 저장
    - 성능 보통, 확장성 보통, 유연성 높음, 복잡성 높음
    - 네오4J

## 1. NoSQL 몽고DB의 특성

- 유연성(Flexibility) : 동적인 스키마 구조
- 확장성(Scalability) : 분산 확장을 고려하여 설계, 수평적 분산 확장
- 고성능(Performance) : 기본적인 읽기와 쓰기 속도가 빠름, 복제와 샤딩을 통한 가용성 향상

## 2. 몽고DB의 구조와 명령문

### 1. 데이터베이스와 컬렉션, 문서

- 문서(Document) : 필드이름과 필드값으로 이루어진 쌍들의 집합, BSON
- 컬렉션(Collection) : 문서들의 모임, `_id` 필드로 각 문서 구분
- 데이터베이스 : 컬렉션들의 모임, 컬렉션들을 구분하는 이름 공간
    - `admin` : 인증과 권한 부여에 관한 데이터 저장
    - `local` : 단일 서버에 대한 데이터 저장
    - `config` : 샤딩된 클러스터에 관한 각 샤드 정보 저장

### 2. 문서 데이터 모델(Document Data Model)

- 데이터의 계층적 표현 가능
- 복잡한 조인 연산 없이도 하나의 문서 안에 모두 표현 가능
- 몽고DB 동적 스키마 : 미리 정해진 고정 스키마가 존재하지 않음
- 몽고DB 관계 표현 : 1:1, 1:N, M:N 관계 표현
    - 내장 방식(Embedded) : 관계를 갖는 데이터를 하나의 문서 안에 함께 저장
    - 참조 방식(Reference) : 관계를 갖는 다른 문서의 키 필드값을 참조키로 저장하는  정규화 형태
- 몽고DB와 관계형 데이터베이스 비교
    - 몽고DB : 데이터베이스, 컬렉션, BSON 문서, BSON 필드, 문서 연결 및 내포, 기본키, 모든 문서가 동일, 참조, 색인
    - 관계형 데이터베이스 : 데이터베이스, 관계, 테이블, 튜플, 행, 레코드, 속성, 열, 조인, 기본키, 테이블마다 다름, 외래키, 색인

# 2. 컬렉션 문서 관리 명령문

## 1. 컬렉션 문서 삽입 Insert문

```sql
db.<컬렉션_이름>.insertOne(<문서1>) # 문서 하나 추가
db.<컬렉션_이름>.insertMany(<문서1>, <문서2>, ...) # 여러 문서 추가
db.<컬렉션_이름>.insert(<문서>|<문서1>, <문서2>, ...) # 문서 하나 또는 여러 개 추가
```

## 2. 컬렉션 문서 검색 Find문

```sql
find({<검색_조건>}, {<검색_필드목록>})
```

- 검색 비교 연산자 : `$eq, $ne, $gt, $gte, $lt, $lte`
- 검색 논리 연산자 : `$and, $or, $in, $nin`
- 검색 문자열 연산자 : `$regex`
- 검색 결과의 정렬, 생략 및 제한 : `sort(), skip(), limit()`

## 3. 컬렉션 문서 수정 Update문

```sql
db.<컬렉션_이름>.updateOne({<수정_검색조건>}, {<수정_옵션>}, {upsert:<논리값>}) # 문서 하나 수정
db.<컬렉션_이름>.updateMany({<수정_검색조건>}, {<수정_옵션>}, {upsert:<논리값>}) # 여러 문서 수정
```

## 4. 데이터베이스, 컬렉션, 문서의 삭제 Drop문, Delete문

```sql
db.<컬렉션_이름>.deleteOne({삭제_검색조건}) # 문서 하나 삭제
db.<컬렉션_이름>.deleteMany({삭제_검색조건}) # 여러 문서 삭제

db.<컬렉션_이름>.drop() # 컬렉션 삭제

db.dropDatabase() # 현재 접속 데이터베이스 삭제
```