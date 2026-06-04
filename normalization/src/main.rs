fn main() {
    println!("======================================================");
    println!(" Day 51-52: 배치 처리 & 정규화 기법 (Rust)");
    println!("======================================================\n");
 
    // 미니배치: [batch=2, features=4]
    let batch = vec![
        vec![1.0, 2.0, 3.0, 4.0], // 샘플 1
        vec![2.0, 4.0, 6.0, 8.0], // 샘플 2
    ];
 
    demo_minibatch(&batch);
    demo_layernorm(&batch);
    demo_batchnorm(&batch);
}

// ----------------------------------------------------------------
// 1. 미니배치와 shape
// ----------------------------------------------------------------
// 데이터를 하나씩이 아니라 여러 개를 [batch, features]로 묶어 처리.
//   - 메모리/연산 효율 (행렬 곱으로 한 번에)
//   - 학습 안정성 (여러 샘플 평균 방향으로 업데이트)
 
fn demo_minibatch(batch: &[Vec<f64>]) {
    println!("-- 1) 미니배치와 shape --\n");
    println!("미니배치 shape = [batch={}, features={}]", batch.len(), batch[0].len());
    for (i, sample) in batch.iter().enumerate() {
        println!("  샘플 {}: {:?}", i + 1, sample);
    }
    println!("-> 여러 샘플을 한 행렬로 묶어 한 번에 처리 (효율 + 안정성)\n");
}

// ----------------------------------------------------------------
// 2. LayerNorm (LLM의 표준 정규화)
// ----------------------------------------------------------------
// 각 '샘플(행)' 안에서 feature들을 평균0 분산1로 정규화 후,
// 학습 가능한 gamma(스케일), beta(이동)로 조정.
//   x_norm = (x - mean) / sqrt(var + eps)
//   out = gamma * x_norm + beta
//
// 왜 LLM은 LayerNorm인가?
//   - 배치 크기에 의존하지 않음 (샘플 하나씩도 정규화 가능)
//   - 시퀀스 길이가 제각각인 언어 데이터에 적합
//   - 각 층 입력 분포를 안정시켜 깊은 신경망 학습을 가능케 함
 
fn layernorm(row: &[f64], gamma: &[f64], beta: &[f64], eps: f64) -> Vec<f64> {
    let n = row.len() as f64;
    let mean = row.iter().sum::<f64>() / n;
    let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let denom = (var + eps).sqrt();
    row.iter()
        .enumerate()
        .map(|(i, &x)| gamma[i] * (x - mean) / denom + beta[i])
        .collect()
}
 
fn demo_layernorm(batch: &[Vec<f64>]) {
    println!("-- 2) LayerNorm (각 샘플을 정규화) --\n");
    let gamma = vec![1.0; 4]; // 스케일 (학습 파라미터, 여기선 1)
    let beta = vec![0.0; 4];  // 이동  (학습 파라미터, 여기선 0)
 
    println!("LayerNorm 출력 (각 행이 평균0 분산1로):");
    for row in batch {
        let out = layernorm(row, &gamma, &beta, 1e-5);
        let mean = out.iter().sum::<f64>() / out.len() as f64;
        let var = out.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / out.len() as f64;
        println!("  {:?}  (평균={:.4}, 분산={:.4})",
            out.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>(), mean, var);
    }
    println!("-> 샘플마다 독립적으로 정규화. 배치 크기와 무관 -> LLM에 적합\n");
}

// ----------------------------------------------------------------
// 3. BatchNorm (비교용) — 정규화 '방향'이 다르다
// ----------------------------------------------------------------
// 각 'feature(열)'를 배치 전체에 걸쳐 정규화.
//   문제: 배치가 작거나 1이면 통계 불안정 -> 시퀀스 데이터에 부적합.
//   그래서 LLM은 BatchNorm 대신 LayerNorm을 쓴다.
 
fn demo_batchnorm(batch: &[Vec<f64>]) {
    println!("-- 3) BatchNorm (각 feature를 정규화) — 비교용 --\n");
    let rows = batch.len();
    let cols = batch[0].len();
    let eps = 1e-5;
 
    // 열별 평균/분산
    let mut out = vec![vec![0.0; cols]; rows];
    for j in 0..cols {
        let mean = (0..rows).map(|i| batch[i][j]).sum::<f64>() / rows as f64;
        let var = (0..rows).map(|i| (batch[i][j] - mean).powi(2)).sum::<f64>() / rows as f64;
        let denom = (var + eps).sqrt();
        for i in 0..rows {
            out[i][j] = (batch[i][j] - mean) / denom;
        }
    }
    println!("BatchNorm 출력:");
    for row in &out {
        println!("  {:?}", row.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
    }
    println!();
    println!("핵심 차이:");
    println!("  LayerNorm: 행(샘플) 방향 정규화 -> 배치 크기 무관 -> LLM 표준");
    println!("  BatchNorm: 열(feature) 방향 정규화 -> 배치에 의존 -> CNN 등에서 사용\n");
}