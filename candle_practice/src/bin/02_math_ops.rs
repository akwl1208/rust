// ================================================================
// candle 예제 02: 수학 연산 (element-wise, 행렬곱, 집계)
//
// 어떨 때 쓰나: 신경망의 거의 모든 계산. 가중합, 활성화 전 단계,
// 손실 계산 등에서 이 연산들을 조합해서 쓴다.
// ================================================================

use candle_core::{Tensor, Device, D};
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 02: 수학 연산 ===\n");
 
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev)?; // 2x3
    let b = Tensor::new(&[[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]], &dev)?; // 2x3
 
    // --- element-wise (원소별) ---
    // 같은 위치끼리 연산. 모양이 같아야 함.
    println!("add (a+b):\n{}", a.add(&b)?);
    println!("sub (a-b):\n{}", a.sub(&b)?);
    println!("mul (a*b, 원소별):\n{}", a.mul(&b)?);
    println!("div (a/b):\n{}", a.div(&b)?);

    // 스칼라 연산: 연산자 오버로딩도 됨 (Result 반환에 주의).
    // affine(mul, add): a*2 + 1. 스케일+이동을 한 번에.
    println!("affine a*2+1:\n{}", a.affine(2.0, 1.0)?);

    // --- 단항 함수 (원소마다 적용) ---
    println!("sqr (제곱):\n{}", a.sqr()?);
    println!("sqrt:\n{}", a.sqrt()?);
    println!("exp:\n{}", a.exp()?);
    println!("log:\n{}", a.log()?);
    // pow: 스칼라 거듭제곱
    println!("powf(3.0):\n{}", a.powf(3.0)?);

    // --- 행렬 곱 (matmul) ---
    // element-wise mul과 다름! (m,k)@(k,n)=(m,n)
    let m1 = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &dev)?; // 2x2
    let m2 = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &dev)?; // 2x2
    println!("\nmatmul (행렬곱):\n{}", m1.matmul(&m2)?);

    // --- 집계 (reduction) ---
    // 전체
    println!("\nsum_all: {}", a.sum_all()?.to_scalar::<f32>()?);
    println!("mean_all: {}", a.mean_all()?.to_scalar::<f32>()?);
    println!("max_all: {}", a.max_all()?.to_scalar::<f32>()?);
    println!("min_all: {}", a.min_all()?.to_scalar::<f32>()?);

    // 축 지정 (D::Minus1 = 마지막 축 = 열 방향 -> 행별 결과)
    println!("\nsum 행별: {:?}", a.sum(D::Minus1)?.to_vec1::<f32>()?);
    println!("mean 행별: {:?}", a.mean(D::Minus1)?.to_vec1::<f32>()?);
    // argmax: 최댓값의 '위치(인덱스)'. 분류에서 예측 클래스 고를 때 필수!
    println!("argmax 행별: {:?}", a.argmax(D::Minus1)?.to_vec1::<u32>()?);

    Ok(())
}