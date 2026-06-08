// ================================================================
// candle 예제 08: 정규화 (normalization) with 브로드캐스트
//
// 어떨 때 쓰나: 데이터의 각 feature(열)를 평균 0, 표준편차 1로 맞출 때.
// 신경망 입력 전처리, BatchNorm/LayerNorm의 핵심 패턴.
// 핵심: (샘플, feature) 데이터에 (feature,) 통계를 브로드캐스트로 적용.
// ================================================================
 
use candle_core::{Tensor, Device, D};
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 08: 정규화 (broadcast) ===\n");
 
    // 4개 샘플, 3개 feature. (4, 3)
    // 열(feature)마다 스케일이 다른 상황을 일부러 만들었다.
    //   feature0: 1~4,  feature1: 10~40,  feature2: 100~400
    let x = Tensor::new(
        &[
            [1.0f32, 10.0, 100.0],
            [2.0,    20.0, 200.0],
            [3.0,    30.0, 300.0],
            [4.0,    40.0, 400.0],
        ],
        &dev,
    )?;
    println!("입력 x (4x3):\n{x}\n");
 
    // --- 1) 열(feature)별 평균 구하기 ---
    // mean(0) = 0번 축(행 방향)을 따라 평균 -> 열마다 하나씩 남음 -> (3,)
    //   keepdim 안 쓰면 (3,), 브로드캐스트엔 (3,)도 잘 동작.
    let mean = x.mean(0)?; // (3,)
    println!("열별 평균 mean: {:?}", mean.to_vec1::<f32>()?);

    // --- 2) 열별 표준편차 구하기 ---
    // 표준편차 = sqrt(mean((x - mean)^2)). candle엔 std 헬퍼가 없어 직접 계산.
    //   (x - mean)을 만들 때 (4,3) - (3,) 라서 broadcast_sub 사용!
    let centered = x.broadcast_sub(&mean)?; // (4,3), 각 행에서 평균 빼기
    let var = centered.sqr()?.mean(0)?;      // (3,) 분산
    let std = var.sqrt()?;                    // (3,) 표준편차
    println!("열별 표준편차 std: {:?}", std.to_vec1::<f32>()?);

    // --- 3) 정규화: (x - mean) / std ---
    // centered는 이미 (4,3). std는 (3,). 또 브로드캐스트로 나눔.
    //   eps를 더해 0으로 나누기 방지 (실전에서 항상 하는 습관).
    let eps = 1e-5;
    let std_safe = (std + eps)?;
    let normalized = centered.broadcast_div(&std_safe)?; // (4,3)
    println!("\n정규화 결과 (각 열이 평균0, 표준편차1):\n{normalized}");

    // 검증: 정규화 후 각 열의 평균은 ~0, 표준편차는 ~1 이어야 함.
    let check_mean = normalized.mean(0)?;
    println!("\n정규화 후 열별 평균(≈0): {:?}", check_mean.to_vec1::<f32>()?);

    Ok(())
}