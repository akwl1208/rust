// ================================================================
// candle 예제 05: 비교, 조건 선택, 브로드캐스트
//
// 어떨 때 쓰나: 마스킹(특정 조건 위치만 골라 처리), ReLU 같은 조건 연산,
// 모양 다른 텐서끼리 연산(편향 더하기 등). Transformer 어텐션 마스크에도.
// ================================================================

use candle_core::{Tensor, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 05: 비교 & 조건 & 브로드캐스트 ===\n");

    let a = Tensor::new(&[1.0f32, 5.0, 2.0, 8.0], &dev)?;
    let b = Tensor::new(&[3.0f32, 3.0, 3.0, 3.0], &dev)?;

    // --- 비교: 결과는 0/1 텐서 ---
    // gt = greater than. a>b 인 위치는 1, 아니면 0.
    let mask = a.gt(&b)?;
    println!("a: {:?}", a.to_vec1::<f32>()?);
    println!("a.gt(b) (a>3?): {:?}  <- 1=참, 0=거짓", mask.to_vec1::<u8>()?);

    // --- where_cond: 조건에 따라 값 선택 ---
    // mask가 1인 위치는 a, 0인 위치는 b. (PyTorch의 torch.where)
    //   언제? "조건 만족하면 이 값, 아니면 저 값" 마스킹.
    //   ReLU도 사실 where(x>0, x, 0)으로 표현 가능.
    let chosen = mask.where_cond(&a, &b)?;
    println!("where(a>3, a, b): {:?}", chosen.to_vec1::<f32>()?);

    // --- 브로드캐스트 ---
    // 모양 다른 텐서끼리 연산. candle은 broadcast_ 접두사를 명시적으로 씀.
    let matrix = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev)?; // 2x3
    let bias = Tensor::new(&[10.0f32, 20.0, 30.0], &dev)?; // (3,)

    // broadcast_add: (2,3) + (3,) -> 각 행에 bias를 더함
    //   언제? 신경망에서 W@x 결과에 편향 b 더할 때. 매우 흔함.
    let out = matrix.broadcast_add(&bias)?;
    println!("\nmatrix + bias (broadcast):\n{out}");

    // 스칼라 브로드캐스트는 affine이나 연산자로
    println!("matrix * 10:\n{}", matrix.affine(10.0, 0.0)?);
    Ok(())
}