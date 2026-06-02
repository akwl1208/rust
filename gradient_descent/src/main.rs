fn main() {
    println!("==================================================");
    println!(" Day 40-41: 손실함수 & 경사하강법");
    println!("==================================================\n");
 
    part1_what_is_loss();
    part2_mse();
    part3_cross_entropy();
}

// ================================================================
// Part 1: 손실함수란?
// ================================================================
//
// 손실함수(Loss Function) = "모델의 예측이 정답과 얼마나 틀렸나"를
// 하나의 숫자로 표현하는 함수.
//
//   손실이 크다  = 많이 틀렸다 (나쁨)
//   손실이 작다  = 잘 맞췄다  (좋음)
//   손실 = 0     = 완벽
//
// 학습의 목표 = 이 손실을 최소화하는 가중치를 찾는 것.
// 문제 종류에 따라 쓰는 손실이 다르다:
//   - 회귀(숫자 예측)   -> MSE
//   - 분류/언어모델     -> Cross-Entropy
 
fn part1_what_is_loss() {
    println!("-- Part 1: 손실함수란? --\n");
    println!("손실함수 = '예측이 정답과 얼마나 틀렸나'를 숫자 하나로");
    println!();
    println!("  손실 큼  = 많이 틀림 (나쁨)");
    println!("  손실 작음 = 잘 맞춤  (좋음)");
    println!("  손실 0   = 완벽");
    println!();
    println!("학습의 목표 = 손실을 최소화하는 가중치 찾기");
    println!("문제별 손실:  회귀 -> MSE,  분류/언어모델 -> Cross-Entropy\n");
}

// ================================================================
// Part 2: MSE (Mean Squared Error) - 회귀의 손실
// ================================================================
//
//      MSE = (1/N) * sum (예측 - 정답)^2
//
//   - 오차를 제곱: 부호 제거 + 큰 오차에 더 큰 벌점
//   - 회귀(숫자를 맞추는 문제)에서 표준으로 쓰임
//   - 예: 집값 예측, 온도 예측
 
fn part2_mse() {
    println!("-- Part 2: MSE - 회귀의 손실 --\n");
    println!("MSE = (1/N) sum (예측 - 정답)^2\n");
 
    // 정답 vs 두 모델의 예측 비교
    let targets = [3.0, 5.0, 7.0, 9.0];
    let good_pred = [3.1, 4.9, 7.2, 8.8]; // 거의 맞음
    let bad_pred = [1.0, 8.0, 4.0, 12.0]; // 많이 틀림
 
    let mse_good = mse_pairs(&good_pred, &targets);
    let mse_bad = mse_pairs(&bad_pred, &targets);
 
    println!("정답:        {:?}", targets);
    println!("좋은 예측:    {:?}  -> MSE = {:.4}", good_pred, mse_good);
    println!("나쁜 예측:    {:?}  -> MSE = {:.4}", bad_pred, mse_bad);
    println!();
    println!("-> 예측이 정답에 가까울수록 MSE가 작다\n");
}

// ================================================================
// Part 3: Cross-Entropy - 분류/언어모델의 손실
// ================================================================
//
// 분류 문제에서 모델은 '각 클래스일 확률'을 출력한다.
// Cross-Entropy는 정답 클래스에 얼마나 높은 확률을 줬는지 평가.
//
//      Loss = -log(정답 클래스의 예측 확률)
//
//   - 정답에 확률 1.0 -> -log(1) = 0       (완벽)
//   - 정답에 확률 0.1 -> -log(0.1) = 2.30  (많이 틀림)
//   - 정답에 확률 -> 0 -> 손실 -> 무한대    (완전 틀림)
//
// LLM은 '다음 토큰 분류기'라서 이 손실로 학습한다. (Day 35 복습)
 
fn part3_cross_entropy() {
    println!("-- Part 3: Cross-Entropy - 분류/언어모델의 손실 --\n");
    println!("Loss = -log(정답 클래스의 예측 확률)\n");
 
    println!("정답 클래스에 준 확률에 따른 손실:");
    println!("{:>12} {:>14}", "정답 확률", "Cross-Entropy");
    println!("{}", "-".repeat(28));
    for p in [0.99, 0.8, 0.5, 0.2, 0.05] {
        let loss = -(p as f64).ln();
        println!("{p:>12.2} {loss:>14.4}");
    }
    println!();
    println!("-> 정답에 높은 확률을 줄수록 손실이 0에 가까움");
    println!("   정답 확률이 낮을수록 손실이 급격히 커짐");
    println!("   LLM = '다음 토큰 분류기' -> 이 손실로 학습\n");
}

// ================================================================
// 헬퍼 함수
// ================================================================
 
/// 실습용 데이터: y = 2x + 1
fn sample_data() -> Vec<(f64, f64)> {
    vec![(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0), (5.0, 11.0)]
}
 
/// 더 큰 데이터(12개): y = 2x + 1
fn sample_data_large() -> Vec<(f64, f64)> {
    (1..=12).map(|i| (i as f64, 2.0 * i as f64 + 1.0)).collect()
}
 
/// MSE = (1/N) sum (w*x + b - y)^2
fn mse(data: &[(f64, f64)], w: f64, b: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|(x, y)| (w * x + b - y).powi(2)).sum::<f64>() / n
}
 
/// 예측 배열 vs 정답 배열의 MSE
fn mse_pairs(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n
}
 
/// MSE의 기울기 (dL/dw, dL/db)
///   dL/dw = (1/N) sum 2(w*x+b - y)*x
///   dL/db = (1/N) sum 2(w*x+b - y)
fn gradient(data: &[(f64, f64)], w: f64, b: f64) -> (f64, f64) {
    let n = data.len() as f64;
    let mut dw = 0.0;
    let mut db = 0.0;
    for (x, y) in data {
        let err = w * x + b - y;
        dw += 2.0 * err * x;
        db += 2.0 * err;
    }
    (dw / n, db / n)
}