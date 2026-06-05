
// ================================================================
// candle 예제 04: 인덱싱, 자르기, 합치기
//
// 어떨 때 쓰나: 배치에서 일부만 떼어내거나, 여러 텐서를 이어붙이거나,
// 특정 위치 값을 뽑을 때. 데이터 전처리/후처리에서 자주.
// ================================================================
 
use candle_core::{Tensor, Device, IndexOp, D};
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 04: 인덱싱 & 합치기 ===\n");
 
    let a = Tensor::arange(0f32, 12f32, &dev)?.reshape((3, 4))?; // 3x4
    println!("a (3x4):\n{a}\n");
 
    // --- 인덱싱 (IndexOp 트레잇 필요) ---
    // i(0) = 0번 행 가져오기. PyTorch의 a[0] 과 같다.
    println!("a.i(0) 0번 행: {:?}", a.i(0)?.to_vec1::<f32>()?);
    // i((.., 1)) = 모든 행의 1번 열. (.. 은 전체)
    println!("a.i((.., 1)) 1번 열: {:?}", a.i((.., 1))?.to_vec1::<f32>()?);

    // --- narrow: 특정 축에서 범위 잘라내기 ---
    // narrow(축, 시작, 길이). 0번 축에서 1번부터 2개 행.
    //   언제? 시퀀스에서 일부 구간만, 배치에서 일부 샘플만 뽑을 때.
    let part = a.narrow(0, 1, 2)?; // 1,2번 행
    println!("\nnarrow(0,1,2) 1~2행:\n{part}");

    // --- cat: 이어붙이기 (concatenate) ---
    // 같은 축 방향으로 텐서들을 연결. 차원 수 유지.
    let b = Tensor::arange(0f32, 4f32, &dev)?.reshape((1, 4))?; // 1x4
    let cat = Tensor::cat(&[&a, &b], 0)?; // 0번 축(행)으로 붙임 -> 4x4
    println!("\ncat([a,b], dim=0): shape={:?}", cat.shape().dims());

    // --- stack: 새 축을 만들며 쌓기 ---
    // cat과 달리 '새 차원'이 생긴다. (4,)짜리 둘 -> (2,4)
    let v1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev)?;
    let v2 = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &dev)?;
    let stacked = Tensor::stack(&[&v1, &v2], 0)?; // 2x4
    println!("stack([v1,v2], 0): shape={:?}", stacked.shape().dims());

    // --- 차이 정리 ---
    // cat   : 기존 축에 이어붙임 (차원 수 그대로)
    // stack : 새 축을 만들어 쌓음 (차원 수 +1)
    println!("\ncat=기존 축에 연결, stack=새 축 만들어 쌓기");

    Ok(())
}