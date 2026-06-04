# candle 텐서 생성 함수 정리

> `Tensor::zeros`, `ones`, `randn`, `from_vec`, `new` —
> 같은 `Tensor::`로 시작하는데 매개변수가 다 다른 이유 정리.

---

## 큰 그림

다섯 함수 모두 **텐서를 새로 만드는** 함수다.
무엇으로 채우느냐에 따라 필요한 정보가 달라서 매개변수가 다르다.

핵심 질문 두 가지로 갈린다:
1. **값을 어디서 가져오나?** (0/1/난수로 채울지, 내 데이터를 쓸지)
2. **shape와 dtype을 내가 말해줘야 하나, candle이 자동으로 아나?**

> 한 줄 규칙: **candle이 스스로 알 수 없는 정보만 내가 넘긴다.**

---

## 함수별 정리

### zeros / ones — "정해진 값으로 채워줘"

```rust
Tensor::zeros((2,3), DType::F32, &device)?
//            shape   dtype       device
```

- 0(또는 1)으로 채운다.
- 값만으론 "몇 행 몇 열인지(shape)", "어떤 자료형인지(dtype)"를 알 수 없다.
- 그래서 **shape와 dtype을 둘 다 직접 지정**해야 한다.
- 매개변수: `(shape, dtype, device)`

### randn — "난수로 채워줘"

```rust
Tensor::randn(0f32, 1f32, (2,3), &device)?
//            평균   표준편차  shape   device
```

- 정규분포 난수로 채운다.
- 앞에 **평균, 표준편차**가 붙는다 (어떤 난수를 만들지 정해야 하니까).
- shape는 지정해야 한다.
- dtype은 **자동** — 평균을 `0f32`로 줬으니 f32인 걸 candle이 안다.
- 매개변수: `(평균, 표준편차, shape, device)`

### from_vec — "내 Vec 데이터로 만들어줘"

```rust
let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
Tensor::from_vec(data, (2,3), &device)?
//               데이터  shape   device
```

- 내가 가진 `Vec` 데이터를 텐서로 바꾼다.
- 값(data)은 내가 주므로 따로 채울 필요 없음.
- shape는 지정해야 한다 (1차원 데이터를 "2행 3열로 배치"하라고 알려줌).
  데이터 개수 == shape 곱 이어야 함 (6개 = 2×3).
- dtype은 **자동** — data가 `Vec<f32>`라 f32.
- 매개변수: `(데이터, shape, device)`

### new — "중첩 배열로 만들어줘"

```rust
Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?
//          중첩배열                        device
```

- 모양이 이미 잡힌 중첩 배열을 준다.
- shape **자동** — `[[1,2],[3,4]]`가 2×2인 게 배열 구조에서 보인다.
- dtype **자동** — `1.0f32`에서 f32.
- 매개변수: `(배열, device)` — 가장 단순.

---

## 한눈 비교표

| 함수 | 값을 어디서 | shape | dtype | 전체 매개변수 |
|------|-----------|-------|-------|------------|
| `zeros` | 0으로 채움 | 직접 지정 | 직접 지정 | (shape, dtype, device) |
| `ones` | 1로 채움 | 직접 지정 | 직접 지정 | (shape, dtype, device) |
| `randn` | 난수로 채움 | 직접 지정 | 자동(평균값) | (평균, 표준편차, shape, device) |
| `from_vec` | 내 Vec에서 | 직접 지정 | 자동(Vec 타입) | (데이터, shape, device) |
| `new` | 중첩 배열에서 | 자동(배열) | 자동(배열) | (배열, device) |

---

## 규칙 요약

candle이 **스스로 알 수 없는 정보만** 내가 넘긴다:

- 값을 안 주면 (zeros/ones/randn) → 채울 방법을 알려줘야 함
- shape를 데이터 모양에서 알 수 없으면 (zeros/randn/from_vec) → shape 지정
- dtype을 값에서 추론할 수 없으면 (zeros/ones) → dtype 지정
- **device는 예외 없이 항상 넘김** — candle은 "어느 장치에 만들지"를
  절대 추측하지 않는다. 그래서 다섯 함수 모두 마지막이 `&device`.

> 자세히 보면 `&device`만 다섯 함수의 공통 매개변수다.
> 나머지는 "내가 안 주면 candle이 모르는 것"만 채워 넣는 패턴.
> 이 규칙을 알면 새 생성 함수를 봐도 매개변수를 예측할 수 있다.
