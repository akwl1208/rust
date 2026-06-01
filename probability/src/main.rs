fn main() {
    println!("========================================");
    println!(" Day 35: 확률·소프트맥스·크로스엔트로피");
    println!("========================================\n");
 
    ex1_probability_basics();
    ex2_softmax();
}

// ────────────────────────────────────────────
// 실습 1: 확률의 기초
// ────────────────────────────────────────────
fn ex1_probability_basics() {
    println!("── 실습 1: 확률 기초 ──\n");
 
    // 확률의 두 가지 조건:
    //   1. 모든 값은 0 이상 1 이하
    //   2. 전체 합은 반드시 1.0
 
    // LLM 다음 토큰 예측 예시
    // "안녕" 다음에 올 수 있는 단어들의 확률
    let tokens = ["하세요", "!", "?", "히", "들"];
    let probs  = [0.60_f64, 0.20, 0.10, 0.07, 0.03];
 
    println!("'안녕' 다음 토큰 확률 분포:");
    println!("{:<8} {:>8} {:>20}", "토큰", "확률", "시각화");
    println!("{}", "-".repeat(40));
    for (token, &p) in tokens.iter().zip(probs.iter()) {
        let bar = "█".repeat((p * 30.0) as usize);
        println!("{:<8} {:>8.2}  {}", token, p, bar);
    }
 
    let sum: f64 = probs.iter().sum();
    println!("\n확률 합계: {sum:.2}  (반드시 1.0)\n");
 
    // 조건부 확률: P(A|B) = "B가 주어졌을 때 A의 확률"
    // LLM에서: P(다음토큰 | 지금까지의 문장)
    println!("조건부 확률 P(A|B) 예시:");
    println!("  P('하세요' | '안녕') = 0.60");
    println!("  P('!'      | '안녕') = 0.20");
    println!("  → LLM은 이걸 계산하는 기계\n");
 
    // 독립 사건: P(A and B) = P(A) × P(B)
    // "안녕 하세요"가 연속으로 나올 확률
    let p_annyeong  = 0.05_f64; // 문장에서 '안녕' 등장 확률
    let p_haseyo    = 0.60_f64; // '안녕' 다음에 '하세요' 확률
    let p_together  = p_annyeong * p_haseyo;
    println!("연속 확률 (chain rule):");
    println!("  P('안녕') = {p_annyeong}");
    println!("  P('하세요'|'안녕') = {p_haseyo}");
    println!("  P('안녕 하세요') = {p_annyeong} × {p_haseyo} = {p_together:.4}");
    println!("  → LLM이 문장 전체 확률을 계산하는 방식\n");
}

// ────────────────────────────────────────────
// 실습 2: 소프트맥스 (Softmax)
// ────────────────────────────────────────────
fn ex2_softmax() {
    println!("── 실습 2: 소프트맥스 ──\n");

    // LLM의 마지막 레이어는 각 토큰에 대한 점수(logit)를 출력합니다.
    // 이 점수는 확률이 아닙니다 — 음수도 있고 합이 1이 아닙니다.
    //
    // Softmax = 이 점수들을 확률로 변환하는 함수
    //
    // 공식: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    //
    // 핵심 성질:
    //   1. 출력값이 모두 0~1 사이
    //   2. 출력값 합계가 정확히 1.0
    //   3. 큰 값은 더 크게, 작은 값은 더 작게 (차이 강조)

    let logits = vec![3.0_f64, 1.0, 0.2];
    let tokens = ["하세요", "!", "?"];

    println!("입력 (logits): {:?}", logits);
    println!("  → 모델이 각 토큰에 준 '점수'. 확률이 아님!\n");

    // Softmax 단계별 계산
    println!("Softmax 계산 과정:");

    // Step 1: exp() 적용
    let exp_vals: Vec<f64> = logits.iter().map(|&x| x.exp()).collect();
    println!("  Step 1: exp 적용");
    for ((&logit, &e), token) in logits.iter().zip(exp_vals.iter()).zip(tokens.iter()) {
        println!("    exp({logit:4.1}) = {e:8.4}  [{token}]");
    }

    // Step 2: 합계
    let sum_exp: f64 = exp_vals.iter().sum();
    println!("  Step 2: 합계 = {sum_exp:.4}");

    // Step 3: 각각 합계로 나누기
    let probs = softmax(&logits);
    println!("  Step 3: 각각 / {sum_exp:.4}");
    println!();

    // 결과 출력
    println!("{:<8} {:>8} {:>10} {:>10} {:>20}",
        "토큰", "logit", "exp(x)", "확률", "시각화");
    println!("{}", "-".repeat(60));
    for i in 0..tokens.len() {
        let bar = "█".repeat((probs[i] * 30.0) as usize);
        println!("{:<8} {:>8.1} {:>10.4} {:>10.4}  {}",
            tokens[i], logits[i], exp_vals[i], probs[i], bar);
    }
    let prob_sum: f64 = probs.iter().sum();
    println!("\n확률 합계: {prob_sum:.6}  ✓\n");

    // 수치 안정성 문제
    println!("⚠️  수치 안정성 문제:");
    let big_logits = vec![1000.0_f64, 999.0, 998.0];
    let naive_exp: Vec<f64> = big_logits.iter().map(|&x| x.exp()).collect();
    println!("  logits = [1000, 999, 998]");
    println!("  exp(1000) = {:?}  ← 오버플로우!", naive_exp[0]);

    // 해결: 최댓값을 빼고 계산 (수학적으로 동일)
    // softmax(x_i) = softmax(x_i - max) 이므로
    let stable_probs = softmax_stable(&big_logits);
    println!("  안정화: 최댓값(1000)을 빼면 [0, -1, -2]");
    println!("  결과: {:?}\n", stable_probs.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
}

// ================================================================
// 헬퍼 함수
// ================================================================

/// Softmax — 기본 버전
fn softmax(x: &[f64]) -> Vec<f64> {
    let exp_vals: Vec<f64> = x.iter().map(|&v| v.exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Softmax — 수치 안정화 버전 (최댓값을 빼고 계산)
/// softmax(x_i - max) 는 softmax(x_i) 와 수학적으로 동일
fn softmax_stable(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Cross-Entropy Loss — L = -log(정답 인덱스의 확률)
fn cross_entropy(probs: &[f64], correct_idx: usize) -> f64 {
    -(probs[correct_idx] + 1e-10).ln() // 1e-10: log(0) = -∞ 방지
}

/// Perplexity 계산용 평균 크로스 엔트로피
fn perplexity_from_probs(probs: &[f64]) -> f64 {
    let n = probs.len() as f64;
    -probs.iter().map(|&p| (p + 1e-10).ln()).sum::<f64>() / n
}