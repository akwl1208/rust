// ================================================================
// Day 55-56 보충: 같은 걸 burn으로 (candle과 비교용)
//
// burn은 백엔드를 '타입 파라미터 <B>'로 받는 게 특징.
// 이 추상화 덕에 CPU/GPU/WebGPU를 타입만 바꿔 교체할 수 있다.
// candle과 달리 연산이 Result가 아니라 바로 Tensor를 반환한다(에러는 panic).
//
// Cargo.toml:
//   [dependencies]
//   burn = { version = "0.18", features = ["ndarray"] }
//
// 실행:  cargo run
// ================================================================

use burn::backend::NdArray;          // CPU 백엔드 (ndarray 기반)
use burn::tensor::{Tensor, TensorData};

// 백엔드 타입 별칭. NdArray = CPU.
// GPU로 바꾸려면 이 줄만 Wgpu 등으로 교체하면 끝.
type B = NdArray;

fn main() {
    // burn의 device는 백엔드의 기본값으로 얻는다.
    let device = Default::default();

    // --- 텐서 생성 ---
    // ::<B, 2> = "백엔드 B에서 2차원(rank 2) 텐서".
    // candle의 Tensor::zeros((2,3), ...) 에 해당.
    let zeros = Tensor::<B, 2>::zeros([2, 3], &device);
    let ones = Tensor::<B, 2>::ones([2, 3], &device);
    println!("zeros dims: {:?}", zeros.dims());
    println!("ones dims: {:?}", ones.dims());

    // 중첩 배열에서 바로 생성 (from_data 가 가장 일반적)
    let a = Tensor::<B, 2>::from_data([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device); // 2x3
    let bmat = Tensor::<B, 2>::from_data([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device); // 3x2

    // Vec 데이터로 만들려면 TensorData를 거친다 (candle의 from_vec 느낌)
    let _from_vec = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        &device,
    );

    // --- 행렬 곱 ---
    // (2x3) @ (3x2) = (2x2).
    // candle은 a.matmul(&b)? 였지만, burn은 ?가 없고 인자를 값으로 받는다.
    let c = a.clone().matmul(bmat);
    println!("matmul dims: {:?}", c.dims());

    // --- 집계 ---
    // 전체 합 -> 스칼라. into_scalar로 일반 숫자로 꺼낸다.
    let total = ones.sum();
    println!("ones 전체 합: {}", total.into_scalar());

    // 스칼라 곱 (연산자 오버로딩 지원)
    let scaled = a.clone() * 2.0;
    println!("a * 2 dims: {:?}", scaled.dims());

    println!("\n-> burn: 백엔드를 타입(<B>)으로 추상화. ?가 없는 대신 제네릭 등장.");
    println!("   candle보다 구조적이지만 처음엔 더 복잡하게 느껴진다.");
    println!("   (작은 실습/추론은 candle, 큰 학습 시스템은 burn이 강점)");
}