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

// ────────────────────────────────────────────
// 실습 2: 수치 미분
// f'(x) ≈ (f(x+h) - f(x-h)) / 2h
// ────────────────────────────────────────────
fn ex2_numerical_diff() {
    println!("── 실습 2: 수치 미분 ──\n");
 
    // 중앙 차분: f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    // h가 작을수록 정확, 너무 작으면 부동소수점 오차
 
    println!("f(x) = x²   해석적 미분: f'(x) = 2x\n");
    println!("{:<6} {:>12} {:>12} {:>12}", "x", "수치미분", "해석미분", "오차");
    println!("{}", "-".repeat(46));

    for &x in &[0.5_f64, 1.0, 2.0, 3.0, -1.5] {
        let num = numerical_diff(|t| t * t ,x);
        let ana = 2.0 * x;
        println!("{:<6.1} {:>12.8} {:>12.8} {:>12.2e}", x, num, ana, (num-ana).abs());
    }

    // h 크기별 정확도
    println!("\nh 크기별 정확도 (x=2.0, 정답=4.0):");
    println!("{:<10} {:>14} {:>12}", "h", "수치미분", "오차");
    println!("{}", "-".repeat(38));
    for exp in [1, 2, 4, 6, 8, 10, 12, 15] {
        let h = 10.0_f64.powi(-exp);
        let res = numerical_diff_h(|t| t * t, 2.0, h);
        println!("{:<10.0e} {:>14.10} {:>12.2e}", h, res, (res - 4.0).abs());
    }
    println!("  → h = 1e-5 ~ 1e-7 구간이 가장 정확\n");

    // 다양한 함수
    println!("다양한 함수 수치 미분 확인:");
    let cases: &[(&str, fn(f64) -> f64, fn(f64) -> f64, f64)] = &[
        ("x³",     |x| x.powi(3), |x| 3.0 * x.powi(2), 2.0),
        ("sin(x)", |x| x.sin(),   |x| x.cos(),           0.0),
        ("exp(x)", |x| x.exp(),   |x| x.exp(),            1.0),
        ("ln(x)",  |x| x.ln(),    |x| 1.0 / x,           2.0),
    ];
    for &(name, f, df, x) in cases {
        let num = numerical_diff(f, x);
        let ana = df(x);
        println!("  {name:<10} at x={x}: 수치={num:.6}  해석={ana:.6}");
    }
    println!();
}

// ================================================================
// 헬퍼 함수
// ================================================================
 
/// 수치 미분 — 중앙 차분 (h=1e-5)
fn numerical_diff<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    numerical_diff_h(f, x, 1e-5)
}
 
/// 수치 미분 — h 직접 지정
fn numerical_diff_h<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}