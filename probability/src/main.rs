fn main() {
    println!("========================================");
    println!(" Day 35: 확률·소프트맥스·크로스엔트로피");
    println!("========================================\n");
 
    ex1_probability_basics();
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
