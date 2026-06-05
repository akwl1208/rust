# candle 주요 기능 정리 (예제와 함께)

> candle-core의 `Tensor`가 제공하는 핵심 기능을 카테고리별로 정리.
> 각 항목은 "언제 쓰나 + 메서드"로 구성. 예제 파일과 1:1 대응.

---

## 예제 파일 구성

| 파일 | 주제 | 핵심 |
|------|------|------|
| `01_creation_shape.rs` | 생성 & shape 조작 | arange, reshape, unsqueeze, squeeze |
| `02_math_ops.rs` | 수학 연산 | add/mul/matmul, sum/mean/argmax |
| `03_activations.rs` | 활성화 함수 | relu, gelu, silu, tanh |
| `04_indexing_combine.rs` | 인덱싱 & 합치기 | i(), narrow, cat, stack |
| `05_compare_broadcast.rs` | 비교 & 조건 & 브로드캐스트 | gt, where_cond, broadcast_add |
| `06_autograd.rs` | **자동미분** | Var, backward, GradStore |
| `07_linear_regression.rs` | **autograd로 학습** | 선형회귀 종합 |

> 굵게 표시한 06, 07이 가장 중요. 손으로 짠 역전파의 자동화 버전.

---

## 1. 생성 (Creation)

| 메서드 | 하는 일 | 언제 |
|--------|---------|------|
| `zeros(shape, dtype, dev)` | 0으로 채움 | 버퍼 초기화 |
| `ones(shape, dtype, dev)` | 1로 채움 | 초기화 |
| `full(v, shape, dev)` | 특정 값으로 채움 | 상수 텐서 |
| `randn(평균, 표준편차, shape, dev)` | 정규분포 난수 | 가중치 초기화 |
| `rand(lo, hi, shape, dev)` | 균등분포 난수 | 초기화 |
| `arange(start, end, dev)` | 0,1,2... 수열 | 인덱스/데이터 생성 |
| `from_vec(vec, shape, dev)` | Vec 데이터로 | 내 데이터 변환 |
| `new(중첩배열, dev)` | 배열 리터럴로 | 작은 텐서 직접 |
| `eye(n, dtype, dev)` | 단위행렬 | 선형대수 |

---

## 2. Shape 조작

| 메서드 | 하는 일 | 언제 |
|--------|---------|------|
| `reshape((a,b))` | 원소 수 유지하며 모양 변경 | 1D↔2D 변환 |
| `unsqueeze(dim)` | 크기1 축 추가 | 배치 차원 추가 |
| `squeeze(dim)` | 크기1 축 제거 | 불필요한 차원 제거 |
| `flatten_all()` | 전부 1차원으로 | FC층 입력 준비 |
| `transpose(d1,d2)` | 두 축 교환 | 행렬 전치 |
| `t()` | 마지막 2축 전치 | 2D 전치 단축 |
| `permute(...)` | 축 순서 재배열 | 다차원 축 정리 |
| `broadcast_as(shape)` | 브로드캐스트로 확장 | 모양 맞추기 |

> shape 조회: `shape().dims()`, `dim(i)`, `rank()`(차원 수)

---

## 3. 수학 연산

### element-wise (원소별, 모양 같아야 함)
`add`, `sub`, `mul`, `div` — 사칙연산
`sqr`(제곱), `sqrt`, `exp`, `log`, `powf(n)`, `recip`(역수), `neg`(부호반전), `abs`

### 스칼라
`affine(mul, add)` — `x*mul + add` 를 한 번에

### 행렬
`matmul(&other)` — 행렬 곱 (element-wise mul과 다름!)
`broadcast_matmul` — 배치 행렬곱

### 집계 (reduction)
| 전체 | 축 지정 | keepdim |
|------|---------|---------|
| `sum_all` | `sum(dim)` | `sum_keepdim(dim)` |
| `mean_all` | `mean(dim)` | `mean_keepdim(dim)` |
| `max_all` | `max(dim)` | `max_keepdim(dim)` |
| `min_all` | `min(dim)` | `min_keepdim(dim)` |

`argmax(dim)` / `argmin(dim)` — 최대/최소값의 **위치**(분류 예측에 필수)

> 축 지정엔 `D::Minus1`(마지막 축)을 자주 쓴다. 차원이 바뀌어도 안전.

---

## 4. 활성화 함수 (내장)

| 메서드 | 함수 | 주 사용처 |
|--------|------|-----------|
| `relu()` | max(0,x) | 딥러닝 기본 |
| `gelu()` | GELU(tanh 근사) | Transformer(GPT/BERT) |
| `gelu_erf()` | GELU(정확) | 정밀 필요 시 |
| `silu()` | x·sigmoid(x) | LLaMA 등 |
| `tanh()` | tanh | RNN 등 |
| `elu(alpha)` | ELU | 대안 활성화 |

> sigmoid는 직접: `(x.neg()?.exp()? + 1.0)?.recip()?`

---

## 5. 인덱싱 & 합치기

| 메서드 | 하는 일 | 언제 |
|--------|---------|------|
| `i(idx)` (IndexOp) | 인덱싱 (PyTorch `a[idx]`) | 행/열/원소 뽑기 |
| `narrow(dim,start,len)` | 범위 잘라내기 | 시퀀스 구간, 배치 슬라이스 |
| `cat(&[..], dim)` | 이어붙이기 | 기존 축에 연결 (차원 유지) |
| `stack(&[..], dim)` | 새 축으로 쌓기 | 차원 +1 |
| `chunk(n, dim)` | n등분 | 분할 처리 |
| `index_select(dim, idx)` | 인덱스로 선택 | 임베딩 조회 |
| `gather(dim, idx)` | 위치별 수집 | 고급 인덱싱 |

> `cat`은 기존 축에 연결(차원 그대로), `stack`은 새 축 생성(차원 +1).

---

## 6. 비교 & 조건

| 메서드 | 하는 일 |
|--------|---------|
| `gt`,`ge`,`lt`,`le`,`eq`,`ne` | 비교 → 0/1 텐서 |
| `where_cond(&a, &b)` | 조건1→a, 0→b 선택 (마스킹) |
| `maximum`,`minimum` | 원소별 최대/최소 |
| `clamp(min,max)` | 범위 제한 |

> 어텐션 마스크, 조건부 연산에 핵심.

---

## 7. 브로드캐스트 (모양 다른 텐서 연산)

candle은 명시적으로 `broadcast_` 접두사 사용:
`broadcast_add`, `broadcast_mul`, `broadcast_sub`, `broadcast_div`, `broadcast_matmul`

> 가장 흔한 용도: 신경망에서 `W@x` 결과에 편향 `b` 더하기.
> numpy는 자동이지만 candle은 의도를 코드에 명시.

---

## 8. 자동미분 (autograd) ★ 핵심

학습의 심장. Day 48-50에서 손으로 짠 역전파의 자동화.

| 요소 | 역할 |
|------|------|
| `Var::new(값, dev)` | 미분 추적되는 텐서 (학습 파라미터) |
| `var.as_tensor()` | Var를 텐서처럼 사용 |
| `tensor.backward()` | 역전파 실행 → `GradStore` 반환 |
| `grads.get(&var)` | 특정 Var의 gradient 꺼내기 |
| `var.set(&new)` | Var 값 갱신 (업데이트) |

**학습 루프 패턴:**
```rust
let grads = loss.backward()?;        // 역전파 (자동!)
let dw = grads.get(&w).unwrap();     // gradient 꺼내기
let new_w = (w.as_tensor() - (dw * lr)?)?;
w.set(&new_w)?;                      // 업데이트
```

> 이게 PyTorch의 `loss.backward()` + `optimizer.step()`과 같은 원리.
> 우리가 직접 미분을 안 짜도 candle이 계산 그래프를 추적해 자동 계산.

---

## 9. 자료형 & 장치

| 메서드 | 하는 일 |
|--------|---------|
| `to_dtype(DType::F64)` | 자료형 변환 (F32→F64, BF16 등) |
| `to_device(&dev)` | 장치 이동 (CPU↔GPU) |
| `dtype()` | 자료형 조회 |
| `device()` | 장치 조회 |

---

## 10. 값 꺼내기 (텐서 → Rust 값)

| 메서드 | 결과 |
|--------|------|
| `to_scalar::<f32>()` | 스칼라(0차원) → 숫자 하나 |
| `to_vec1::<f32>()` | 1차원 → `Vec<f32>` |
| `to_vec2::<f32>()` | 2차원 → `Vec<Vec<f32>>` |
| `to_vec3::<f32>()` | 3차원 → 중첩 Vec |

> 타입 파라미터(`::<f32>`)로 어떤 자료형으로 꺼낼지 지정.

---

## 11. 헷갈리기 쉬운 개념 (직접 공부하며 막혔던 것들)

### (1) 같은 변수명을 `let`으로 또 쓰는 것 = 섀도잉(shadowing)

```rust
let a = Tensor::arange(0f32, 6f32, &dev)?;  // 첫 번째 a
let a = a.reshape((2,3))?;                   // let 또 씀 → 새로운 a
```

`let`은 값 변경을 막는데 왜 `a`를 또 쓸까? → **이건 값 변경이 아니라
"같은 이름으로 새 변수를 만드는 것"** 이다. 이를 섀도잉이라 한다.

- `let a = 5; a = 10;` → ❌ 값 변경 (금지)
- `let a = 5; let a = 10;` → ✅ 섀도잉 (새 변수, 허용)
- `let mut a = 5; a = 10;` → ✅ 진짜 값 변경 (mut일 때만)

비유: 칠판의 "a" 이름표.
- 값 변경 = 그 자리 내용을 고쳐 쓰기 (let이 금지)
- 섀도잉 = 기존 이름표 위에 새 "a" 이름표를 덮어 붙이기 (허용)

**왜 쓰나**: 같은 데이터를 단계적으로 변형할 때 이름을 유지하면 편하다.
```rust
let a = Tensor::arange(...)?;  // 1차원
let a = a.reshape((2,3))?;      // 2x3으로
let a = a.t()?;                 // 전치
```
보너스: 섀도잉은 **타입도 바꿀 수 있다**(새 변수라서). mut은 타입 고정.

---

### (2) sum의 "축(axis)" — 어느 방향으로 더하나

`a = [[0,1,2],[3,4,5]]` (2×3) 일 때:

**축 0 (세로, 행 방향)으로 더하기** — 위아래로 합침:
```
[[0, 1, 2],
 [3, 4, 5]]
 ↓  ↓  ↓
[3, 5, 7]      (0+3, 1+4, 2+5) → "열별 합", 결과 3개
```

**축 1 (가로, 열 방향)으로 더하기** — 좌우로 합침:
```
[[0, 1, 2],  → 0+1+2 = 3
 [3, 4, 5]]  → 3+4+5 = 12   → "행별 합", 결과 2개
```

> 핵심: "어느 축으로 더하느냐"에 따라 더해서 없어지는 방향이 정해진다.
> 축 0으로 더하면 행이 사라지고(열별 합), 축 1로 더하면 열이 사라진다(행별 합).

---

### (3) `D::Minus1` — "마지막 축"을 차원 수와 무관하게 가리킴

축에는 0부터 번호가 붙고, **마지막 축의 번호는 차원 수에 따라 다르다**:
- 2차원: 축 0,1 → 마지막 = 1
- 3차원: 축 0,1,2 → 마지막 = 2
- 4차원: 축 0,1,2,3 → 마지막 = 3

숫자로 직접 쓰면(`a.sum(1)`) 2차원일 때만 마지막이라,
차원이 바뀌면 일일이 고쳐야 한다(버그 위험).

`D::Minus1`은 숫자가 아니라 **"무조건 맨 마지막 축"** 이라는 뜻:
```rust
a.sum(D::Minus1)?   // 차원이 2든 3든 4든 항상 마지막 축으로 더함
```
→ 코드를 안 고쳐도 항상 맞는 축을 가리킨다. (파이썬 `list[-1]`의 -1과 같은 발상)

**왜 자주 쓰나**: 딥러닝 데이터는 보통 `[배치, ..., 특징]` 모양이라
"마지막 축 = 특징"인 경우가 많다. "각 샘플의 특징을 합/평균" 같은 연산이
곧 마지막 축 연산이라 `D::Minus1`이 자주 등장한다.

---

## 핵심 요약

- candle의 거의 모든 기능은 `Tensor`의 **메서드**로 제공된다.
- 거의 모든 연산이 `Result`를 반환 → `?`로 처리.
- 다른 텐서를 인자로 줄 땐 `&`로 빌림.
- 모양 다른 연산은 `broadcast_` 명시.
- **학습의 핵심은 autograd**: `Var` + `backward()`로 역전파 자동화.
- 전체 메서드 목록은 docs.rs의 `Tensor` 페이지에서 확인.