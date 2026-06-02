fn main() {
    println!("==================================================");
    println!(" Day 40-41: 손실함수 & 경사하강법");
    println!("==================================================\n");
 
    part1_what_is_loss();
    part2_mse();
    part3_cross_entropy();
    part4_gradient_descent();
    part5_learning_rate_experiment();   // 핵심 실습 1
    part6_batch_vs_sgd();               // 핵심 실습 2
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
// Part 4: 경사하강법 (Gradient Descent)
// ================================================================
//
// 손실을 줄이는 가중치를 어떻게 찾나? -> 기울기를 따라 내려간다.
//
//   비유: 안개 낀 산에서 골짜기로 내려가기
//     - 발 밑 경사(기울기)를 느낀다
//     - 가장 가파른 내리막 방향으로 한 걸음
//     - 반복하면 골짜기(최저 손실)에 도착
//
//   업데이트 공식:
//     가중치 <- 가중치 - 학습률 * 기울기
//
//   기울기(gradient) = 손실이 가장 빨리 '커지는' 방향
//   -> 그 반대(-)로 가면 손실이 '줄어든다'
 
fn part4_gradient_descent() {
    println!("-- Part 4: 경사하강법 --\n");
    println!("비유: 안개 낀 산에서 골짜기로 내려가기");
    println!("  1) 발밑 경사(기울기)를 느낀다");
    println!("  2) 가장 가파른 내리막으로 한 걸음");
    println!("  3) 반복 -> 골짜기(최저 손실) 도착");
    println!();
    println!("공식:  가중치 <- 가중치 - 학습률 * 기울기");
    println!();
 
    // y = 2x + 1 데이터로 한 스텝씩 직접 보여주기
    let data = sample_data();
    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.01;
 
    println!("y=2x+1 데이터, w=0,b=0 에서 시작 (lr=0.01):");
    println!("{:>6} {:>10} {:>10} {:>12}", "step", "w", "b", "MSE");
    println!("{}", "-".repeat(40));
    for step in 0..=5 {
        if step > 0 {
            let (dw, db) = gradient(&data, w, b);
            w -= lr * dw;
            b -= lr * db;
        }
        println!("{step:>6} {w:>10.4} {b:>10.4} {:>12.4}", mse(&data, w, b));
    }
    println!("-> 한 걸음마다 MSE가 줄어든다 (산을 내려가는 중)\n");
}

// ================================================================
// Part 5: 학습률 실험 (★ 핵심 실습)
// ================================================================
//
// 학습률(Learning Rate) = 한 걸음의 크기.
//
//   너무 작으면 (0.001): 한 걸음이 너무 작아 영원히 못 내려감 (느림)
//   적당하면   (0.01):   적절히 내려가 골짜기 도착 (좋음)
//   너무 크면  (0.1):    한 걸음이 너무 커서 골짜기를 건너뛰고
//                        반대편 벽으로 튀어오름 -> 점점 발산 (폭발)
//
// 같은 데이터/시작점에서 학습률만 바꿔 결과를 비교한다.
// (적정 학습률은 데이터 스케일에 따라 다르다 - 여기선 이 데이터 기준)
 
fn part5_learning_rate_experiment() {
    println!("-- Part 5: 학습률 실험 (0.001 vs 0.01 vs 0.1) --\n");
 
    let data = sample_data();
    let report_at = [1, 10, 50, 200, 1000];
 
    for &lr in &[0.001, 0.01, 0.1] {
        let label = match lr {
            x if x < 0.005 => "너무 작음 (느림)",
            x if x < 0.05 => "적정 (좋음)",
            _ => "너무 큼 (발산 위험)",
        };
        println!("== 학습률 lr = {lr}  [{label}] ==");
        println!("{:>6} {:>14} {:>14} {:>16}", "step", "w", "b", "MSE");
        println!("{}", "-".repeat(54));
 
        let mut w = 0.0;
        let mut b = 0.0;
        for step in 1..=1000 {
            let (dw, db) = gradient(&data, w, b);
            w -= lr * dw;
            b -= lr * db;
            if report_at.contains(&step) {
                let m = mse(&data, w, b);
                let m_str = if m.is_nan() || m.is_infinite() || m > 1e10 {
                    "발산(폭발)".to_string()
                } else {
                    format!("{m:.6}")
                };
                println!("{step:>6} {w:>14.4} {b:>14.4} {m_str:>16}");
            }
        }
        println!();
    }
 
    println!("결론:");
    println!("  lr=0.001: 1000걸음을 가도 도착 못함 (보폭이 너무 작음)");
    println!("  lr=0.01 : 깔끔하게 골짜기 도착 (w~2, b~1)");
    println!("  lr=0.1  : 골짜기를 건너뛰고 점점 멀어짐 -> 숫자 폭발(발산)");
    println!("  => 학습률은 '너무 작지도 크지도 않게' 맞추는 게 핵심\n");
}

// ================================================================
// Part 6: Batch vs Mini-batch vs SGD (★ 핵심 실습)
// ================================================================
//
// 기울기를 '데이터 몇 개로' 계산하느냐의 차이.
//
//   [Batch GD] 전체 데이터로 기울기 계산 후 한 번 업데이트
//     + 안정적이고 정확한 방향
//     - 데이터 많으면 한 걸음이 너무 느리고 무거움
//
//   [SGD (Stochastic)] 데이터 1개로 기울기 계산, 매번 업데이트
//     + 매우 빠르고 자주 업데이트
//     - 방향이 들쭉날쭉(노이즈 많음) -> 하지만 그 덕에 얕은 골짜기 탈출도
//
//   [Mini-batch] 일부(예: 4개, 32개)로 계산 -> 둘의 절충
//     + 적당히 안정적 + 적당히 빠름 + GPU 효율 좋음
//     => 실무 딥러닝의 표준 (보통 32~256개)
//
// 세 방식 모두 결국 같은 골짜기로 수렴한다. 가는 길이 다를 뿐.
 
fn part6_batch_vs_sgd() {
    println!("-- Part 6: Batch vs Mini-batch vs SGD --\n");
 
    println!("차이 = '기울기를 데이터 몇 개로 계산하느냐'");
    println!("  Batch     : 전체로 계산 -> 안정적이나 무거움");
    println!("  SGD        : 1개로 계산  -> 빠르나 방향이 들쭉날쭉");
    println!("  Mini-batch : 일부로 계산 -> 절충 (실무 표준, 보통 32~256)");
    println!();
 
    let data = sample_data_large(); // 12개 데이터
    let lr = 0.005;
    let epochs = 200;
 
    // Batch GD
    let (bw, bb) = {
        let (mut w, mut b) = (0.0, 0.0);
        for _ in 0..epochs {
            let (dw, db) = gradient(&data, w, b);
            w -= lr * dw;
            b -= lr * db;
        }
        (w, b)
    };
 
    // SGD (1개씩, 순서대로) - seed 없이 순차로 단순화
    let (sw, sb) = {
        let (mut w, mut b) = (0.0, 0.0);
        for _ in 0..epochs {
            for pt in &data {
                let (dw, db) = gradient(std::slice::from_ref(pt), w, b);
                w -= lr * dw;
                b -= lr * db;
            }
        }
        (w, b)
    };
 
    // Mini-batch (4개씩)
    let (mw, mb) = {
        let (mut w, mut b) = (0.0, 0.0);
        for _ in 0..epochs {
            for chunk in data.chunks(4) {
                let (dw, db) = gradient(chunk, w, b);
                w -= lr * dw;
                b -= lr * db;
            }
        }
        (w, b)
    };
 
    println!("12개 데이터(y=2x+1), {epochs} epoch, lr={lr} 결과:");
    println!("{:>14} {:>10} {:>10} {:>12}", "방식", "w", "b", "MSE");
    println!("{}", "-".repeat(48));
    println!("{:>14} {bw:>10.4} {bb:>10.4} {:>12.6}", "Batch", mse(&data, bw, bb));
    println!("{:>14} {sw:>10.4} {sb:>10.4} {:>12.6}", "SGD(1개)", mse(&data, sw, sb));
    println!("{:>14} {mw:>10.4} {mb:>10.4} {:>12.6}", "Mini-batch(4)", mse(&data, mw, mb));
    println!();
    println!("-> 세 방식 모두 같은 정답(w~2,b~1)으로 수렴한다.");
    println!("   '가는 길'이 다를 뿐 도착지는 같다.");
    println!("   SGD/미니배치는 업데이트가 잦아 큰 데이터에서 훨씬 빠르다.\n");
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