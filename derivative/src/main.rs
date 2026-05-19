fn main() {
    println!("========================================");
    println!(" Day 33-34: 미분·기울기·편미분");
    println!("========================================\n");
 
    ex1_derivative_intuition();
    ex2_numerical_diff();
}

// ────────────────────────────────────────────
// 실습 1: 미분의 직관 — "얼마나 민감한가"
// ────────────────────────────────────────────
fn ex1_derivative_intuition() {
    println!("── 실습 1: 미분의 직관 ──\n");

    // f(x) = x²  →  f'(x) = 2x
    // 의미: x를 Δx만큼 바꾸면 f(x)는 약 f'(x)·Δx만큼 바뀐다

    let x = 3.0_f64;
    let delta = 0.001_f64;

    let fx = x * x;
    let fx_delta = (x + delta) * (x + delta);
    let actual_change = fx_delta - fx;
    let predicted = 2.0 * x * delta; // f'(x) · Δx

    println!("f(x) = x²   at x = {x}");
    println!("  f({x})           = {fx}");
    println!("  f({x} + {delta}) = {fx_delta:.6}");
    println!("  실제 변화량      = {actual_change:.6}");
    println!("  미분으로 예측    = 2×{x}×{delta} = {predicted:.6}");
    println!("  → 거의 일치! 미분 = '변화 예측기'\n");

    // x별 기울기 시각화
    println!("x별 기울기 f'(x) = 2x:");
    for &xi in &[-2.0_f64, -0.1, 0.0, 1.0, 2.0, 3.0] {
        let slope = 2.0 * xi;
        let bar_len = (slope.abs() as usize).min(15);
        let bar = if slope >= 0.0 {
            format!("{:>15}|{}", "", "▶".repeat(bar_len))
        } else {
            format!("{:>15}{}", "◀".repeat(bar_len), "|")
        };
        println!("  x={xi:5.1}  f'={slope:5.1}  {bar}");
    }
    println!();
}

