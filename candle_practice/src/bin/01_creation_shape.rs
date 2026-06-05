use candle_core::{Tensor, DType, Device, D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 01: 생성 & shape 조작 ===\n");

    // arange: 0,1,2,...,n-1 수열 (numpy의 arange). 데이터 만들 때 편리.
    let a = Tensor::arange(0f32, 6f32, &dev)?;
    println!("arange(0..6): {:?}", a.to_vec1::<f32>()?);
 
    // reshape: 원소 수 유지하며 모양 바꾸기. (6,) -> (2,3)
    //   언제? 1차원 데이터를 행렬로 만들 때, 배치 형태로 맞출 때.
    let a = a.reshape((2,3))?;
    println!("reshape (2,3):\n{a}");

    // full: 특정 값으로 가득 채우기
    let f = Tensor::full(7f32, (2,2), &dev)?;
    println!("full(7): \n{f}");

    // eye: 단위행렬(대각선 1). 선형대수에서 자주.
    let e = Tensor::eye(3, DType::F32, &dev)?;
    println!("eye(3):\n{e}");

    // --- shape 조작 ---
    // unsqueeze: 크기 1짜리 축 추가. (2,3) -> (1,2,3)
    //   언제? 배치 차원 추가할 때. 모델은 보통 [batch, ...] 형태를 원함.
    let b = a.unsqueeze(0)?;
    println!("\nunsqueeze(0): shape={:?}", b.shape().dims());

    // squeeze: 크기 1짜리 축 제거. (1,2,3) -> (2,3)
    let c = b.squeeze(0)?;
    println!("squeeze(0): shape={:?}", c.shape().dims());

    // flatten_all: 전부 1차원으로 펴기. (2,3) -> (6,)
    //   언제? CNN 출력을 fully-connected 층에 넣기 전 등.
    let flat = a.flatten_all()?;
    println!("flatten_all: {:?}", flat.to_vec1::<f32>()?);

    // transpose / t: 축 교환. t()는 마지막 두 축 전치(2D면 행<->열).
    println!("\n전치 t():\n{}", a.t()?);

    // --- 차원 지정에 쓰는 D ---
    // D::Minus1 = 마지막 축. 차원 수가 바뀌어도 "마지막"을 안전하게 가리킴.
    let s = a.sum(D::Minus1)?; // 마지막 축(열)으로 합 -> 행별 합
    println!("\nsum(D::Minus1) 행별 합: {:?}", s.to_vec1::<f32>()?);

    Ok(())
}
