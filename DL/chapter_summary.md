# DL 01~04 챕터별 핵심 & 헷갈리는 것 정리

---

## DL_01 | Python / numpy 기초

### 핵심 개념
- **브로드캐스트**: 형태가 다른 배열끼리 연산할 때 numpy가 자동으로 크기를 맞춰줌
  ```python
  a = np.array([[1,2],[3,4]])  # (2,2)
  b = np.array([3,4])          # (2,) → 자동으로 (2,2)처럼 동작
  a + b  # [[4,6],[6,8]]
  ```
- **행렬곱 차원 규칙**: `(m,n) · (n,k) = (m,k)` 안쪽이 같아야 함
- **열 순회**: `zip(*x)` 또는 인덱스로 열 방향 접근

### 헷갈리는 것
| 상황 | 주의 |
|------|------|
| `astype(np.float32)` | float32는 정밀도가 낮아 `11.2000001` 같은 오차 발생 |
| `ndim` vs `shape` vs `size` | ndim=차원수, shape=형태튜플, size=전체원소수 |
| `reshape` vs `transpose` | reshape=원소 재배치, transpose=행↔열 전치 |
| `np.dot(v1, v2)` | 벡터끼리면 내적(스칼라), 행렬이면 행렬곱 |
| `arange` vs `linspace` | arange=간격 지정, linspace=개수 지정 |
| `np.sort(A)` vs `A.sort()` | `np.sort`는 새 배열 반환, `A.sort()`는 제자리 정렬 |

---

## DL_02 | 퍼셉트론

### 핵심 개념
- **퍼셉트론 구조**: 입력 × 가중치 합산 → 임계값(theta) 또는 편향(bias)과 비교 → 출력
  ```python
  # theta 방식
  if w1*x1 + w2*x2 > theta: return 1
  # bias 방식 (동일)
  if np.sum(x*w) + b > 0: return 1
  ```
- **XOR = 다층 퍼셉트론**: NAND → OR → AND 조합으로 구현
  ```
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y  = AND(s1, s2)
  ```

### 헷갈리는 것
| 상황 | 주의 |
|------|------|
| 단일 퍼셉트론의 한계 | XOR은 직선 하나로 분리 불가 → 다층 필요 |
| theta vs bias | `w1x1+w2x2 > theta` = `w1x1+w2x2+b > 0` (b = -theta), 완전히 동일 |
| NAND 진리표 | (1,1)일 때만 0, 나머지는 1 |

---

## DL_03 | 신경망 / 활성화 함수

### 핵심 개념
- **활성화 함수 종류 및 공식**
  ```python
  sigmoid  : 1 / (1 + np.exp(-x))
  ReLU     : np.maximum(0, x)
  tanh     : 2 / (1 + np.exp(-x)) - 1
  softplus : np.log(1 + np.exp(x))
  step     : np.where(x >= 0, 1, 0)
  ```
- **sigmoid vs step**: sigmoid는 연속·미분 가능, step은 불연속·미분 불가
- **sigmoid a값**: a가 커질수록 step function에 수렴
- **softmax**: 출력 합 = 1, 대소관계 유지, 오버플로 방지로 최댓값 빼기 필수
  ```python
  a -= np.max(a)   # 필수!
  y = np.exp(a) / np.sum(np.exp(a))
  ```

### 헷갈리는 것
| 상황 | 주의 |
|------|------|
| `step1(배열)` | if문은 단일 bool만 판단 → 원소 여러 개면 ValueError |
| `plt.plot(y)` x 생략 | x축이 인덱스(0,1,...,999)로 표시됨 → `plt.plot(x, y)` 로 써야 함 |
| 선형 vs 비선형 | 은닉층에 선형 함수 쓰면 층을 쌓는 의미 없음 → 반드시 비선형 사용 |
| softmax 최댓값 안 빼면 | `np.exp(큰수)` → `inf` → `NaN` 발생 |
| `np.maximum` vs `max` | `np.maximum(0,x)` = 배열 원소별 비교, `max(0,x)` = 스칼라만 가능 |

---

## DL_04 | 신경망 학습 / 손실 함수

### 핵심 개념
- **손실 함수 공식**
  ```python
  # MSE
  E = 1/2 * np.sum((y - t)**2)

  # CEE (delta로 log(0) 방지)
  E = -np.sum(t * np.log(y + 1e-7))

  # 미니배치 CEE (N으로 나눠 정규화)
  E = -1/N * np.sum(np.sum(t * np.log(y + delta)))
  ```
- **배치 CEE 구현 2가지**
  ```python
  # cross_entropy_error1: 원-핫 인코딩 t일 때
  -np.sum(t * np.log(y + 1e-17)) / batch_size

  # cross_entropy_error2: 정답 레이블 인덱스 t일 때 (계산 효율적)
  -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
  ```
- **수치 미분 2가지**
  ```python
  # 전진 차분 (오차 큼)
  (f(x+h) - f(x)) / h

  # 중앙 차분 (오차 작음, 권장)
  (f(x+h) - f(x-h)) / (2h)  # 분모 주의: h가 아니라 2h
  ```

### 헷갈리는 것
| 상황 | 주의 |
|------|------|
| CEE에 delta 안 넣으면 | `log(0) = -inf` → 계산 불가 |
| `y.ndim == 1` 체크 | 이미지 1장(1D)과 배치(2D)를 같은 함수로 처리하기 위한 분기 |
| `t.reshape(1, t.size)` | 1D 배열을 2D로 바꿔 배치 처리와 형태 통일 |
| `y[np.arange(batch_size), t]` | 각 행에서 정답 인덱스 위치의 값만 추출하는 fancy indexing |
| 정확도 대신 손실 함수 쓰는 이유 | 정확도는 미분값이 거의 0 → 학습 불가. 손실 함수는 연속·미분 가능 |
| 전진 차분 vs 중앙 차분 | 중앙 차분이 오차 더 작음. 분모가 `h`가 아니라 `2h` (코드에선 `/h` 쓰는 경우 많으니 확인) |
