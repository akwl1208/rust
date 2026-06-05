// candle의 핵심 타입 3개를 가져온다:
//   Tensor = 다차원 배열 (numpy의 ndarray에 해당)
//   DType  = 원소의 자료형 (F32, F64, ...)
//   Device = 연산을 수행할 장치 (Cpu, Cuda)
use candle_core::{Tensor, DType, Device};

// main이 Result를 반환한다. 이게 핵심 포인트 중 하나:
//   candle의 거의 모든 연산은 실패할 수 있어서 Result<Tensor>를 돌려준다.
//   (shape가 안 맞거나, 장치가 없거나 등)
//   그래서 각 연산 끝에 ? 를 붙여 "성공하면 값 꺼내고, 실패하면 즉시 반환"한다.
//   main이 Result를 반환해야 ?를 main에서도 쓸 수 있다.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Day 55-56: candle Hello Tensor ===\n");
 
    // ============================================================
    // 1. Device — 연산을 어디서 할까 (CPU vs GPU)
    // ============================================================
    // 모든 텐서는 '어느 장치에 사는지'를 가진다.
    // CPU에 사는 텐서와 GPU에 사는 텐서는 바로 연산할 수 없다.
    // (numpy엔 없던 개념. PyTorch의 .to("cuda")와 같은 맥락)

    // 지금은 CPU 사용 (Cargo.toml에 cuda 기능 미적용).
    // GPU(CUDA)를 쓰려면: Cargo.toml에 features=["cuda"] 추가 + nvcc 설치 후
    //   let (device, device_name) = match Device::new_cuda(0) {
    //       Ok(gpu) => (gpu, "CUDA GPU"),
    //       Err(_) => (Device::Cpu, "CPU"),
    //   };
    let device = Device::Cpu;
    println!("[1] Device = CPU\n");

    // ============================================================
    // 2. 텐서 생성 — zeros, ones, randn, from_vec, new
    // ============================================================
    println!("[2] 텐서 생성");
 
    // zeros: 0으로 채운 텐서. (shape, dtype, device) 순서.
    //   shape (2,3) = 2행 3열. dtype F32 = 32비트 실수.
    //   끝에 ? : 생성도 실패할 수 있으므로 Result를 풀어준다.
    let z = Tensor::zeros((2,3), DType::F32, &device)?;
    println!("  zeros (2x3):\n{z}");

    // ones: 1로 채운 텐서.
    let o = Tensor::ones((2,3), DType::F32, &device)?;
    println!("  ones (2x3):\n{o}");

    // randn: 정규분포(평균 0, 표준편차 1) 난수로 채운 텐서.
    //   첫 두 인자(0f32, 1f32) = 평균, 표준편차.
    let r = Tensor::randn(0f32, 1f32, (2,3), &device)?;
    println!("  randn (2x3):\n{r}");

    // from_vec: 내가 가진 Vec 데이터로 텐서 만들기.
    //   데이터 길이와 shape의 곱이 일치해야 한다 (6개 = 2*3).
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data, (2,3), &device)?;
    println!("  from_vec (2x3):\n{t}");

    // new: 중첩 배열(슬라이스)에서 바로 만들기. shape는 자동 추론.
    let n = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    println!("  new (2x2):\n{n}\n");

    // ============================================================
    // 3. Shape & DType — 텐서의 모양과 자료형
    // ============================================================
    println!("[3] Shape & DType");
    // shape: 텐서의 차원 구조. dims()로 [usize] 슬라이스를 얻는다.
    println!("  t의 shape = {:?}", t.shape().dims());
    println!("  t의 dtype = {:?}", t.dtype());
 
    // dtype 변환: to_dtype 으로 F32 -> F64 같은 변환.
    //   bf16(bfloat16)은 LLM에서 메모리 절약용으로 흔히 쓰는 16비트 실수.
    let t64 = t.to_dtype(DType::F64)?;
    println!("  F64로 변환 후 dtype = {:?}\n", t64.dtype());

    // ============================================================
    // 4. 기본 연산 — add, mul (여기서 소유권이 보인다!)
    // ============================================================
    println!("[4] 기본 연산: add, mul");
 
    // 핵심: 연산 메서드는 인자를 '빌려서(&)' 받는다.
    //   t.add(&o)  <- &o : o를 '빌려준다'. 소유권을 넘기지 않는다.
    //   왜? &를 안 쓰면 o의 소유권이 add로 넘어가 버려서,
    //       그 뒤로 o를 다시 못 쓰게 된다. &로 빌려주면 o는 그대로 살아있다.
    //   이게 "왜 이걸 clone/빌림 해야 하나"의 출발점.
    let sum = t.add(&o)?; // t + o (각 원소에 1 더하기)
    println!("  t + ones:\n{sum}");

    let prod = t.mul(&t)?;  // t * t (각 원소 제곱, element-wise)
    println!("  t * t (제곱):\n{prod}");

     // 스칼라 연산: affine(곱, 더하기). t*2 + 0 = 모든 원소 2배.
     let scaled = t.affine(2.0, 0.0)?;
     println!("  t * 2:\n{scaled}\n");

    // ============================================================
    // 5. 행렬 곱 (matmul) & 전치 (transpose)
    // ============================================================
    println!("[5] matmul & transpose");
 
    // matmul: 진짜 행렬 곱. (2x3) @ (3x2) = (2x2).
    //   element-wise인 mul과 완전히 다르다!
    //   a의 열 수(3) == b의 행 수(3) 여야 한다. 안 맞으면 여기서 에러(?로 처리).
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?; // 2x3
    let b = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device)?; // 3x2
    let c = a.matmul(&b)?; // 2x2
    println!("  a(2x3) @ b(3x2) = c(2x2):\n{c}");

    // transpose: 두 차원을 맞바꾼다. (2,3) -> (3,2).
    //   transpose(0, 1) = 0번 축과 1번 축을 교환.
    let at = a.transpose(0, 1)?;
    println!("  a 전치 (3x2): shape={:?}\n", at.shape().dims());

    // ============================================================
    // 6. 집계 연산 — sum, mean (축 지정이 핵심)
    // ============================================================
    println!("[6] sum & mean");
 
    // sum_all / mean_all: 모든 원소를 다 더하거나 평균.
    //   결과는 스칼라(0차원 텐서). to_scalar로 일반 숫자로 꺼낸다.
    let total = a.sum_all()?;
    println!("  a 전체 합 = {}", total.to_scalar::<f32>()?);

    let avg = a.mean_all()?;
    println!("  a 전체 평균 = {}", avg.to_scalar::<f32>()?);

    // 축(axis) 지정 sum: 특정 차원으로만 더한다.
    //   sum(0) = 0번 축(행 방향)으로 더해 -> 열별 합 (길이 3)
    //   sum(1) = 1번 축(열 방향)으로 더해 -> 행별 합 (길이 2)
    let col_sum = a.sum(0)?; // 열별 합
    println!("  열별 합 (sum axis=0): {:?}", col_sum.to_vec1::<f32>()?);

    let row_sum = a.sum(1)?; // 행별 합
    println!("  행별 합 (sum axis=1): {:?}", row_sum.to_vec1::<f32>()?);

    // keepdim 버전: 더한 축을 1로 '남겨둔다' (차원 수 유지).
    //   브로드캐스트(아래)에 쓰기 좋다.
    let row_sum_kd = a.sum_keepdim(1)?; // shape (2,1)
    println!("  행별 합 keepdim: shape={:?}\n", row_sum_kd.shape().dims());

    // ============================================================
    // 7. 브로드캐스트 — 모양이 다른 텐서끼리 연산
    // ============================================================
    println!("[7] 브로드캐스트");
 
    // numpy에선 그냥 +로 됐지만, candle은 모양이 다르면
    // 'broadcast_' 붙은 메서드를 명시적으로 써야 한다.
    //   a (2x3) 에 행벡터 (1x3) 을 더하기:
    let bias = Tensor::new(&[[10.0f32, 20.0, 30.0]], &device)?; // 1x3
    let a_plus_bias = a.broadcast_add(&bias)?; // 각 행에 [10,20,30] 더함
    println!("  a + bias(broadcast):\n{a_plus_bias}");
    println!("  (numpy의 자동 브로드캐스트를 candle은 명시적으로 표현)\n");
 
    println!("=== 끝. numpy와 거의 같지만 ?(에러)와 &(빌림)이 핵심 차이 ===");
    
    Ok(())
}
