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
    
    Ok(())
}
