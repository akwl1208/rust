fn main() {
    println!("========================================");
    println!(" Day 35: 확률·소프트맥스·크로스엔트로피");
    println!("========================================\n");
 
    ex1_probability_basics();
    ex2_softmax();
    ex3_softmax_temperature();
    ex4_cross_entropy();
    ex5_log_likelihood();
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

// ────────────────────────────────────────────
// 실습 3: 소프트맥스 Temperature
// ────────────────────────────────────────────
fn ex3_softmax_temperature() {
    println!("── 실습 3: Temperature — 확률 분포 조절 ──\n");

    // LLM 추론 시 temperature 파라미터가 있습니다.
    // softmax(x / T) 로 계산합니다.
    //
    // T < 1.0 → 확률 차이가 커짐 → 확실한 답만 고름 (결정적)
    // T = 1.0 → 기본값
    // T > 1.0 → 확률 차이가 줄어듦 → 다양한 답 가능 (창의적)

    let logits = vec![3.0_f64, 1.0, 0.2];
    let tokens = ["하세요", "!", "?"];
    let temperatures = [0.3_f64, 0.7, 1.0, 1.5, 2.0];

    println!("logits = {:?}\n", logits);
    println!("{:<6} {:>10} {:>10} {:>10}  설명",
        "T", tokens[0], tokens[1], tokens[2]);
    println!("{}", "-".repeat(65));

    for &t in &temperatures {
        let scaled: Vec<f64> = logits.iter().map(|&x| x / t).collect();
        let probs = softmax_stable(&scaled);
        let desc = match t as i32 {
            0 => "거의 항상 '하세요'만 선택",
            1 => "기본값",
            2 => "다양한 토큰 선택 가능",
            _ => if t < 1.0 { "결정적" } else { "창의적" },
        };
        println!("T={t:<5} {:>10.4} {:>10.4} {:>10.4}  {desc}",
            probs[0], probs[1], probs[2]);
    }
    println!();

    println!("실제 LLM 사용 예:");
    println!("  temperature=0.1 → 코드 생성, 번역 (정확성 중요)");
    println!("  temperature=1.0 → 일반 대화");
    println!("  temperature=1.5 → 창작, 소설 쓰기 (다양성 중요)\n");
}

// ────────────────────────────────────────────
// 실습 4: 크로스 엔트로피 (Cross-Entropy)
// ────────────────────────────────────────────
fn ex4_cross_entropy() {
    println!("── 실습 4: 크로스 엔트로피 ──\n");

    // 크로스 엔트로피 = "예측이 정답과 얼마나 다른가"
    //
    // 공식: L = -log(정답 토큰의 예측 확률)
    //
    // 직관:
    //   정답 확률이 1.0 → L = -log(1.0) = 0.0    (완벽)
    //   정답 확률이 0.5 → L = -log(0.5) = 0.693  (나쁨)
    //   정답 확률이 0.1 → L = -log(0.1) = 2.303  (매우 나쁨)
    //   정답 확률이 0.0 → L = -log(0.0) = ∞      (완전 틀림)

    println!("L = -log(정답 토큰의 확률)");
    println!("  확률이 높을수록 Loss가 작음\n");

    println!("{:>10} {:>12}", "정답 확률", "Cross-Entropy Loss");
    println!("{}", "-".repeat(25));
    for &p in &[1.0_f64, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01] {
        let loss = -p.ln();
        let bar = "▓".repeat((loss * 5.0).min(30.0) as usize);
        println!("{:>10.2} {:>12.4}  {bar}", p, loss);
    }
    println!();

    // 실제 예시: 문장 "안녕 하세요"
    // 모델이 '하세요'를 얼마나 잘 예측했는가?

    println!("예시: 정답 토큰 = '하세요'\n");

    // 케이스 A: 잘 학습된 모델
    let logits_good = vec![3.0_f64, 0.5, 0.1];
    let probs_good  = softmax_stable(&logits_good);
    let loss_good   = cross_entropy(&probs_good, 0); // 정답 인덱스 = 0
    println!("케이스 A (잘 학습된 모델):");
    println!("  logits = {:?}", logits_good);
    println!("  probs  = [{:.4}, {:.4}, {:.4}]",
        probs_good[0], probs_good[1], probs_good[2]);
    println!("  '하세요' 확률 = {:.4}", probs_good[0]);
    println!("  Loss = -log({:.4}) = {loss_good:.4}\n", probs_good[0]);

    // 케이스 B: 학습 초기 (모든 logit이 비슷)
    let logits_bad = vec![0.1_f64, 0.05, 0.08];
    let probs_bad  = softmax_stable(&logits_bad);
    let loss_bad   = cross_entropy(&probs_bad, 0);
    println!("케이스 B (학습 초기, 랜덤에 가까움):");
    println!("  logits = {:?}", logits_bad);
    println!("  probs  = [{:.4}, {:.4}, {:.4}]",
        probs_bad[0], probs_bad[1], probs_bad[2]);
    println!("  '하세요' 확률 = {:.4}", probs_bad[0]);
    println!("  Loss = -log({:.4}) = {loss_bad:.4}", probs_bad[0]);
    println!();

    println!("Loss 비교: {loss_good:.4} (A)  vs  {loss_bad:.4} (B)");
    println!("학습 = Loss를 A처럼 줄여나가는 과정\n");
}

// ────────────────────────────────────────────
// 실습 5: 로그 가능도 (Log-Likelihood)
// ────────────────────────────────────────────
fn ex5_log_likelihood() {
    println!("── 실습 5: 로그 가능도 ──\n");

    // LLM은 문장 전체의 확률을 최대화하도록 학습합니다.
    //
    // 문장 "안녕 하세요" = 토큰 3개
    // P(문장) = P('안녕') × P('하세요'|'안녕') × P('<끝>'|'안녕 하세요')
    //
    // 문제: 확률을 계속 곱하면 숫자가 너무 작아짐
    //   0.6 × 0.8 × 0.7 × 0.5 × ... → 0.000000001
    //
    // 해결: log를 취하면 곱셈 → 덧셈
    //   log(P(문장)) = log(P1) + log(P2) + log(P3) + ...

    println!("문장 확률 계산: '안녕 하세요'");
    println!("P(문장) = P('안녕') × P('하세요'|'안녕') × P('<끝>'|...)\n");

    // 각 토큰의 예측 확률 (학습된 모델 기준)
    let token_probs = vec![
        ("안녕",  0.05_f64),
        ("하세요", 0.60),
        ("<끝>",  0.80),
    ];

    // 직접 곱하면
    let joint_prob: f64 = token_probs.iter().map(|(_, p)| p).product();
    println!("직접 곱셈:");
    let mut display = String::new();
    for (i, (token, p)) in token_probs.iter().enumerate() {
        if i > 0 { display.push_str(" × "); }
        display.push_str(&format!("P({token})={p}"));
    }
    println!("  {display}");
    println!("  = {joint_prob:.6}  ← 토큰이 많아지면 0에 수렴\n");

    // 로그 가능도
    let log_likelihood: f64 = token_probs.iter()
        .map(|(_, p)| p.ln())
        .sum();
    println!("로그 가능도 (log 적용 → 곱 → 합):");
    for (token, p) in &token_probs {
        println!("  log({p}) = {:.4}  [{token}]", p.ln());
    }
    println!("  합계 = {log_likelihood:.4}\n");

    println!("관계: log_likelihood = log({joint_prob:.6}) = {:.4}",
        joint_prob.ln());
    println!("  → 같은 값! log는 단조증가 함수이므로");
    println!("    최대 가능도 = 최소 (-log 가능도) = 최소 크로스 엔트로피\n");

    // 학습 목표 연결
    println!("학습 목표:");
    println!("  최대화: log P(문장)     ← 로그 가능도 최대화");
    println!("  = 최소화: -log P(문장)  ← 크로스 엔트로피 최소화");
    println!("  → 둘은 같은 말!\n");

    // 학습 과정 시뮬레이션
    println!("학습 진행에 따른 로그 가능도 변화:");
    println!("{:<8} {:>12} {:>12}", "학습단계", "정답확률", "Log-Likelihood");
    println!("{}", "-".repeat(35));
    let stages = [
        ("초기",  0.33_f64),  // 3개 토큰 랜덤 = 1/3
        ("10%",   0.45),
        ("50%",   0.65),
        ("90%",   0.80),
        ("완료",  0.95),
    ];
    for (stage, p) in stages {
        // 3토큰 문장 전체 로그 가능도 근사
        let ll = 3.0 * p.ln();
        println!("{stage:<8} {p:>12.2} {ll:>12.4}");
    }
    println!("  → 학습 완료로 갈수록 0에 가까워짐 (log(1)=0이 최댓값)\n");

    //문장의 그럴듯함은 토큰 확률의 곱인데, 곱은 컴퓨터에서 0으로 터지니까 
    //log로 덧셈을 만들고, 그 값을 최대화하는 게 곧 cross-entropy를 최소화하는 LLM 학습 그 자체다.
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