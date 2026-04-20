# numpy / matplotlib 치트시트 (DL 01~04 기준)

---

## 괄호 규칙 핵심

```
np.zeros((2,2))   ← (()) 쌍괄호: zeros, ones만 해당. shape이 튜플이라 두 겹
A.shape           ← 괄호 없음: 속성
plt.show()        ← 빈 괄호: 메서드는 항상 ()
```

---

## numpy 배열 생성

| 함수 | 예시 | 설명 |
|------|------|------|
| `np.array()` | `np.array([1,2,3])` | 리스트로 배열 생성 |
| `np.zeros((m,n))` | `np.zeros((2,2))` | 0으로 채운 배열, shape은 튜플 |
| `np.ones((m,n))` | `np.ones((2,2))` | 1로 채운 배열, shape은 튜플 |
| `np.arange(start,stop,step)` | `np.arange(0,20,0.01)` | 간격으로 배열 생성 |
| `np.linspace(start,stop,num)` | `np.linspace(-5,5,1000)` | 개수로 균등 분포 배열 |
| `np.random.rand(m,n)` | `np.random.rand(10,10)` | 균등분포 난수, **0 이상 1 미만** |
| `np.random.randn(m,n)` | `np.random.randn(2,3)` | 정규분포 난수, **음수 포함** (평균 0, 표준편차 1) |
| `np.random.seed(n)` | `np.random.seed(44)` | 난수 시드 고정 |

> `np.random` 은 `import numpy as np` 하면 자동으로 딸려옴. 별도 import 불필요.
> `randn` 의 `n` = **normal(정규분포)**. `rand` 는 0~1만, `randn` 은 음수도 나옴.

---

## numpy 수학 함수

| 함수 | 예시 | 설명 |
|------|------|------|
| `np.exp(a)` | `np.exp(-x)` | e^x |
| `np.log(a)` | `np.log(y + 1e-7)` | ln(x), log(0) 방지로 delta 추가 |
| `np.sum(a)` | `np.sum(x*w)` | 합계 |
| `np.maximum(a,b)` | `np.maximum(0,x)` | 원소별 최댓값 (ReLU) |
| `np.where(조건,a,b)` | `np.where(x>=0,1,0)` | 조건 참→a, 거짓→b |
| `np.dot(a,b)` | `np.dot(x,w)` | 내적 / 행렬곱 |
| `np.sin(x)` | `np.sin(x)` | 사인 |
| `np.cos(x)` | `np.cos(x)` | 코사인 |
| `np.ndim(a)` | `np.ndim(a)` | 차원 수 반환 (함수 버전) |
| `np.transpose(a)` | `np.transpose(N)` | 전치 (행↔열) |
| `np.sort(a)` | `np.sort(N)` | 정렬 |

---

## numpy 속성 (괄호 없음)

| 속성 | 설명 |
|------|------|
| `A.shape` | 형태 튜플 ex) `(3,4)` |
| `A.ndim` | 차원 수 |
| `A.size` | 전체 원소 수 |
| `A.dtype` | 데이터 타입 ex) `float64`, `int64` |

---

## numpy 메서드 (괄호 있음)

> `A.std()` 와 `np.std(A)` 는 결과가 같음. numpy는 편의상 둘 다 지원.
> `mean`, `sum`, `max`, `min`, `sort` 도 마찬가지.

| 메서드 | np. 버전 | 설명 |
|--------|----------|------|
| `A.reshape(m,n)` | — | 형태 변경 |
| `A.astype(타입)` | — | 타입 변환 |
| `A.mean()` | `np.mean(A)` | 평균 |
| `A.std()` | `np.std(A)` | 표준편차 |
| `A.max()` / `A.min()` | `np.max(A)` / `np.min(A)` | 최대 / 최소 |
| `A.sum()` | `np.sum(A)` | 합계 |
| `A.sort()` | `np.sort(A)` | 정렬 |

---

## numpy linalg

| 함수 | 설명 |
|------|------|
| `np.linalg.inv(A)` | 역행렬 |
| `np.linalg.det(A)` | 행렬식 |
| `np.linalg.eig(A)` | 고유값, 고유벡터 |
| `np.linalg.matrix_rank(A)` | 랭크 |
| `np.linalg.norm(A)` | 노름 (벡터/행렬 크기) |
| `np.linalg.solve(A,b)` | 연립방정식 Ax=b |

---

## matplotlib

| 함수 | 예시 | 설명 |
|------|------|------|
| `plt.plot(x,y)` | `plt.plot(x, y, label='sin')` | 선 그래프, x 빠뜨리면 인덱스가 x축 |
| `plt.figure(figsize=(w,h))` | `plt.figure(figsize=(8,5))` | 그림 크기(인치), 그래프 그리기 전에 호출 |
| `plt.show()` | `plt.show()` | 그래프 출력 |
| `plt.xlabel()` / `plt.ylabel()` | `plt.xlabel('x')` | 축 레이블 |
| `plt.title()` | `plt.title('sin&cos')` | 제목 |
| `plt.legend()` | `plt.legend()` | 범례 (plot에 label 지정했을 때) |
| `plt.ylim(a,b)` | `plt.ylim(-1,2)` | y축 범위 |
| `plt.grid(True)` | `plt.grid(True)` | 격자 표시 |

### plot 스타일 fmt string
```python
plt.plot(x, y, 'k--')       # 검정 + 점선
plt.plot(x, y, linestyle='--')  # 선스타일만, 색상 자동
# 형식: '색상 + 마커 + 선스타일'  예) 'ro-' → 빨강+원형마커+실선
```

---

## 활성화 함수 / 손실 함수

```python
# Step
np.where(x >= 0, 1, 0)

# Sigmoid
1 / (1 + np.exp(-x))

# ReLU
np.maximum(0, x)

# Softplus
np.log(1 + np.exp(x))

# tanh
2 / (1 + np.exp(-x)) - 1

# MSE
1/2 * np.sum((y - t)**2)

# CEE (log(0) 방지 delta 필수)
-np.sum(t * np.log(y + 1e-7))

# Softmax (오버플로 방지: 최댓값 빼기 필수)
a -= np.max(a)
np.exp(a) / np.sum(np.exp(a))
```

---

## 행렬곱 규칙

```
(m, n) · (n, k) = (m, k)   ← 안쪽 두 수가 반드시 같아야 함
```
안 맞으면 `.reshape()` 또는 `np.transpose()` 로 맞추기

---

## 한눈에 보는 외우기 팁

| 헷갈리는 것 | 기억법 |
|-------------|--------|
| `zeros((2,2))` 쌍괄호 | shape 자체가 튜플 → 튜플을 인수로 넣으니 두 겹. **zeros/ones만 해당** |
| `arange` vs `linspace` | arange = **간격**, linspace = **개수** |
| `np.` vs `A.` | np. = 창고에서 꺼내기, A. = A한테 시키기 |
| 속성 vs 메서드 | 읽기만 → 괄호 없음 / 실행 → 괄호 있음 |
| `plt.plot(x, y)` x 빠뜨리면 | x축이 0,1,2... 인덱스로 표시됨 |
| softmax 최댓값 빼기 | 안 빼면 `inf` → `NaN` 발생 |
