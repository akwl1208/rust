fn main() {
    println!("================================================");
    println!(" Day 36-37: 미니 수학 라이브러리 (from scratch)");
    println!("================================================\n");
 
    part1_matrix();           // 행렬: 곱·전치
    part2_softmax();          // 소프트맥스
    part3_cross_entropy();    // 크로스 엔트로피
    part4_numerical_diff();   // 수치 미분 (Day 36-37의 핵심 새 개념)
    part5_chain_rule();       // 연쇄 법칙 + 수치 검증
    part6_mini_forward();     // 전체 연결: 선형층 → softmax → CE
    run_tests();              // 검산 (손계산 값과 비교)
}

// ================================================================
// Part 1: 행렬 — 곱(matmul)과 전치(transpose)
// ================================================================
//
// 행렬은 Vec<Vec<f64>> 로 표현합니다. (바깥 = 행, 안쪽 = 열)
//
//   A = [[1, 2, 3],     ← 2행 3열 (2×3)
//        [4, 5, 6]]
//
// 행렬 곱 규칙: (m×k) · (k×n) = (m×n)
//   왼쪽 행렬의 '열 수' 와 오른쪽 행렬의 '행 수' 가 같아야 함.
//   결과의 (i,j) 원소 = A의 i번째 행 · B의 j번째 열 (내적)
 
fn part1_matrix() {
    println!("── Part 1: 행렬 곱 & 전치 ──\n");

    // A: 2×3,  B: 3×2  →  결과 C: 2×2
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    let b = vec![
        vec![7.0,  8.0],
        vec![9.0,  10.0],
        vec![11.0, 12.0],
    ];    

    println!("A ({}×{}):", a.len(), a[0].len());
    print_matrix(&a);
    println!("B ({}×{}):", b.len(), b[0].len());
    print_matrix(&b);
 
    let c = matmul(&a, &b);
    println!("A · B ({}×{}):", c.len(), c[0].len());
    print_matrix(&c);
 
    // 손으로 검산:
    //   C[0][0] = 1·7 + 2·9 + 3·11 = 7 + 18 + 33 = 58
    //   C[0][1] = 1·8 + 2·10 + 3·12 = 8 + 20 + 36 = 64
    //   C[1][0] = 4·7 + 5·9 + 6·11 = 28 + 45 + 66 = 139
    //   C[1][1] = 4·8 + 5·10 + 6·12 = 32 + 50 + 72 = 154
    println!("손계산: C = [[58, 64], [139, 154]]");
    println!("→ 코드 결과와 일치하는지 확인!\n");
 
    // 전치: 행과 열을 뒤바꿈 (m×n → n×m)
    let at = transpose(&a);
    println!("A의 전치 Aᵀ ({}×{}):", at.len(), at[0].len());
    print_matrix(&at);
    println!("→ A의 (i,j) 가 Aᵀ의 (j,i) 로 이동\n");
}

// ================================================================
// Part 2: 소프트맥스 (Day 35 복습 + 라이브러리화)
// ================================================================
//
// logit(점수) → 확률 분포로 변환.
//   softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
// 큰 logit에서 오버플로우를 막기 위해 최댓값을 빼고 계산.
 
fn part2_softmax() {
    println!("── Part 2: 소프트맥스 ──\n");
 
    let logits = vec![3.0, 1.0, 0.2];
    let probs = softmax(&logits);
 
    println!("logits = {:?}", logits);
    print!("softmax = [");
    for (i, p) in probs.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{p:.4}");
    }
    println!("]");
    println!("합계 = {:.6}  (반드시 1.0)\n", probs.iter().sum::<f64>());
}

// ================================================================
// Part 3: 크로스 엔트로피 (Day 35 복습 + 라이브러리화)
// ================================================================
//
// 정답이 one-hot 일 때: L = -log(정답 토큰의 예측 확률)
 
fn part3_cross_entropy() {
    println!("── Part 3: 크로스 엔트로피 ──\n");
 
    let logits = vec![3.0, 1.0, 0.2];
    let probs = softmax(&logits);
    let target = 0; // 정답 = 0번 토큰
 
    let loss = cross_entropy(&probs, target);
    println!("probs = [{:.4}, {:.4}, {:.4}]", probs[0], probs[1], probs[2]);
    println!("정답 인덱스 = {target},  확률 = {:.4}", probs[target]);
    println!("Loss = -log({:.4}) = {:.4}\n", probs[target], loss);
}

// ================================================================
// Part 4: 수치 미분 (★ Day 36-37의 핵심 새 개념)
// ================================================================
//
// 미분 = "입력을 아주 조금 바꾸면 출력이 얼마나 바뀌나" (= 기울기)
//
// 정의:  f'(x) = lim(h→0) [f(x+h) - f(x)] / h
//
// 컴퓨터는 h를 0으로 보낼 수 없으니, 아주 작은 h를 대입해 근사합니다.
// 이를 '수치 미분(numerical differentiation)' 이라 합니다.
//
// 더 정확한 방법: 중심차분 (central difference)
//   f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
//   → 앞뒤로 똑같이 보기 때문에 오차가 훨씬 작음.
 
fn part4_numerical_diff() {
    println!("── Part 4: 수치 미분 (★ 새 개념) ──\n");
 
    // 예제 1: f(x) = x²  →  해석적 미분 f'(x) = 2x
    let f = |x: f64| x * x;
 
    println!("f(x) = x²,  해석적 미분 f'(x) = 2x\n");
    println!("{:>5} {:>14} {:>14} {:>10}", "x", "수치 미분", "정답(2x)", "오차");
    println!("{}", "-".repeat(46));
    for &x in &[1.0, 2.0, 3.0, 5.0] {
        let approx = numerical_diff(f, x);
        let exact = 2.0 * x;
        println!("{x:>5.1} {approx:>14.6} {exact:>14.6} {:>10.2e}",
            (approx - exact).abs());
    }
    println!("→ h가 작을수록 정답에 근접 (중심차분 사용)\n");
 
    // 예제 2: 편미분 — 여러 변수 중 하나만 살짝 흔들기
    //   f(x, y) = x²·y
    //   ∂f/∂x = 2xy,   ∂f/∂y = x²
    println!("편미분: f(x,y) = x²·y  at (x=2, y=3)");
    let g = |v: &[f64]| v[0] * v[0] * v[1];
    let point = vec![2.0, 3.0];
    let grad = numerical_gradient(&g, &point);
 
    println!("  ∂f/∂x ≈ {:.4}  (정답 2xy = 2·2·3 = 12)", grad[0]);
    println!("  ∂f/∂y ≈ {:.4}  (정답 x²  = 2²    = 4)", grad[1]);
    println!("  gradient = {:?}",
        grad.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>());
    println!("→ gradient = '어느 방향으로 가야 함수가 가장 빨리 커지나'\n");
}

// ================================================================
// Part 5: 연쇄 법칙 (Chain Rule) — 역전파의 핵심
// ================================================================
//
// 합성 함수의 미분:  y = f(g(x)) 이면
//   dy/dx = f'(g(x)) · g'(x)
//
// 역전파(backprop)는 이 규칙을 수억 번 반복하는 것뿐입니다.
// 여기서는 손으로 계산한 연쇄 법칙 결과를, 수치 미분으로 검증합니다.
 
fn part5_chain_rule() {
    println!("── Part 5: 연쇄 법칙 + 수치 검증 ──\n");
 
    // 예제: y = (3x + 1)²
    //   바깥 함수 f(u) = u²       → f'(u) = 2u
    //   안쪽 함수 g(x) = 3x + 1   → g'(x) = 3
    //   연쇄 법칙: dy/dx = 2(3x+1) · 3 = 6(3x+1)
    let composed = |x: f64| {
        let u = 3.0 * x + 1.0; // g(x)
        u * u                   // f(u)
    };
 
    println!("y = (3x + 1)²");
    println!("연쇄 법칙: dy/dx = 2(3x+1)·3 = 6(3x+1)\n");
    println!("{:>5} {:>14} {:>16} {:>10}", "x", "수치 미분", "연쇄법칙 정답", "오차");
    println!("{}", "-".repeat(48));
    for &x in &[0.0, 1.0, 2.0] {
        let approx = numerical_diff(composed, x);
        let exact = 6.0 * (3.0 * x + 1.0); // 손계산 공식
        println!("{x:>5.1} {approx:>14.6} {exact:>16.6} {:>10.2e}",
            (approx - exact).abs());
    }
    println!("→ 손으로 푼 연쇄 법칙과 수치 미분이 일치!");
    println!("  이 일치를 컴퓨터가 자동으로 하는 게 autograd / 역전파\n");
}

// ================================================================
// Part 6: 전체 연결 — 미니 순전파 (선형층 → softmax → CE)
// ================================================================
//
// LLM 한 스텝의 축소판입니다:
//   입력 x  →  W·x + b (선형층, 행렬 곱)  →  logits
//          →  softmax  →  확률  →  cross-entropy  →  Loss
//
// 그리고 "W를 살짝 바꾸면 Loss가 얼마나 바뀌나" 를 수치 미분으로 확인합니다.
// 이게 바로 다음 주(Week 6) 경사하강법이 사용하는 정보입니다.
 
fn part6_mini_forward() {
    println!("── Part 6: 미니 순전파 (선형층→softmax→CE) ──\n");
 
    // 입력 벡터 (특징 2개)
    let x = vec![1.0, 2.0];
 
    // 가중치 W: 3개 출력(클래스) × 2개 입력 = 3×2 행렬
    let w = vec![
        vec![0.5, -0.3],
        vec![0.1,  0.8],
        vec![-0.4, 0.2],
    ];
    let b = vec![0.0, 0.0, 0.0]; // 편향
    let target = 1; // 정답 클래스 = 1번
 
    // 순전파를 하나의 클로저로 묶기 (Loss 반환)
    let forward = |w: &[Vec<f64>]| {
        let logits = linear(w, &b, &x); // W·x + b
        let probs = softmax(&logits);
        cross_entropy(&probs, target)
    };
 
    let logits = linear(&w, &b, &x);
    let probs = softmax(&logits);
    let loss = cross_entropy(&probs, target);
 
    println!("입력 x = {:?}", x);
    println!("logits (W·x+b) = {:?}",
        logits.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>());
    println!("확률 = {:?}",
        probs.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
    println!("정답 클래스 = {target},  Loss = {loss:.4}\n");
 
    // W의 각 원소를 살짝 흔들어 Loss의 기울기 구하기 (수치 미분)
    println!("∂Loss/∂W (수치 미분, 각 가중치의 기울기):");
    let grad_w = numerical_gradient_matrix(&forward, &w);
    for (i, row) in grad_w.iter().enumerate() {
        let formatted: Vec<String> = row.iter().map(|v| format!("{v:>8.4}")).collect();
        println!("  행{i}: [{}]", formatted.join(", "));
    }
    println!("\n→ 이 기울기의 반대 방향으로 W를 조금씩 옮기면 Loss가 줄어듦");
    println!("  W_new = W - 학습률 × 기울기   ← 이것이 경사하강법 (Week 6)\n");
}

// ================================================================
// 검산 (Tests) — 손으로 계산한 값과 코드 결과 비교
// ================================================================
 
fn run_tests() {
    println!("── 검산 (assert로 자동 확인) ──\n");
    let mut passed = 0;
    let mut total = 0;
 
    // helper: 거의 같은지 (부동소수점 오차 허용)
    fn close(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }
 
    // Test 1: 행렬 곱
    total += 1;
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let bmat = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
    let c = matmul(&a, &bmat);
    if close(c[0][0], 58.0) && close(c[0][1], 64.0)
        && close(c[1][0], 139.0) && close(c[1][1], 154.0) {
        println!("  [PASS] 행렬 곱 = [[58,64],[139,154]]");
        passed += 1;
    } else {
        println!("  [FAIL] 행렬 곱 결과: {:?}", c);
    }
 
    // Test 2: softmax 합 = 1
    total += 1;
    let probs = softmax(&[3.0, 1.0, 0.2]);
    if close(probs.iter().sum::<f64>(), 1.0) {
        println!("  [PASS] softmax 합계 = 1.0");
        passed += 1;
    } else {
        println!("  [FAIL] softmax 합계 = {}", probs.iter().sum::<f64>());
    }
 
    // Test 3: cross-entropy, 정답확률=1 이면 loss≈0
    total += 1;
    let perfect = vec![1.0, 0.0, 0.0];
    let loss = cross_entropy(&perfect, 0);
    if loss < 1e-4 {
        println!("  [PASS] CE(정답확률=1) ≈ 0  (실제 {:.6})", loss);
        passed += 1;
    } else {
        println!("  [FAIL] CE = {}", loss);
    }
 
    // Test 4: 수치 미분  f(x)=x²,  f'(3)=6
    total += 1;
    let d = numerical_diff(|x| x * x, 3.0);
    if (d - 6.0).abs() < 1e-4 {
        println!("  [PASS] d/dx(x²) at x=3 ≈ 6  (실제 {:.6})", d);
        passed += 1;
    } else {
        println!("  [FAIL] 수치 미분 = {}", d);
    }
 
    // Test 5: 편미분  f(x,y)=x²y at (2,3) → [12, 4]
    total += 1;
    let grad = numerical_gradient(&|v: &[f64]| v[0] * v[0] * v[1], &[2.0, 3.0]);
    if (grad[0] - 12.0).abs() < 1e-3 && (grad[1] - 4.0).abs() < 1e-3 {
        println!("  [PASS] ∇(x²y) at (2,3) ≈ [12, 4]  (실제 [{:.3}, {:.3}])",
            grad[0], grad[1]);
        passed += 1;
    } else {
        println!("  [FAIL] gradient = {:?}", grad);
    }
 
    // Test 6: 연쇄 법칙  y=(3x+1)², dy/dx at x=2 = 6·7 = 42
    total += 1;
    let dc = numerical_diff(|x| { let u = 3.0 * x + 1.0; u * u }, 2.0);
    if (dc - 42.0).abs() < 1e-3 {
        println!("  [PASS] d/dx(3x+1)² at x=2 ≈ 42  (실제 {:.4})", dc);
        passed += 1;
    } else {
        println!("  [FAIL] 연쇄 법칙 = {}", dc);
    }
 
    // Test 7: 전치 두 번 = 원본
    total += 1;
    let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let mtt = transpose(&transpose(&m));
    if m == mtt {
        println!("  [PASS] (Aᵀ)ᵀ = A");
        passed += 1;
    } else {
        println!("  [FAIL] 전치 두 번이 원본과 다름");
    }
 
    println!("\n결과: {passed}/{total} 통과");
    if passed == total {
        println!("모든 검산 통과! 미니 라이브러리 완성 ✓");
    }
    println!();
}

// ================================================================
// 라이브러리 함수들 (numpy 없이 직접 구현)
// ================================================================

type Matrix = Vec<Vec<f64>>;

/// 행렬 곱: (m×k) · (k×n) = (m×n)
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let m = a.len();        // A의 행 수
    let k = a[0].len();     // A의 열 수 = B의 행 수
    let n = b[0].len();     // B의 열 수
    assert_eq!(k, b.len(), "shape 불일치: A의 열 수 != B의 행 수");

    let mut c = vec![vec![0.0;n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i][p] * b[p][j]; // i행 · j열 내적
            }
            c[i][j] = sum;
        }
    }
    c
}

/// 전치: (m×n) → (n×m), (i,j) → (j,i)
fn transpose(a: &[Vec<f64>]) -> Matrix {
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// 소프트맥스 (수치 안정화: 최댓값을 빼고 계산)
fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// 크로스 엔트로피: L = -log(정답 인덱스의 확률)
fn cross_entropy(probs: &[f64], target: usize) -> f64 {
    -(probs[target] + 1e-10).ln() // log(0) = -∞ 방지용 작은 값
}

/// 수치 미분 (중심차분): f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
fn numerical_diff<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    let h = 1e-5;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// 수치 기울기(벡터 입력): 각 변수마다 하나씩 편미분
fn numerical_gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64]) -> Vec<f64> {
    let h = 1e-5;
    let mut grad = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * h);
    }
    grad
}

/// 선형층: y = W·x + b
///   W: (출력 × 입력) 행렬,  x: 입력 벡터,  b: 편향 벡터
fn linear(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            let dot: f64 = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            dot + bias
        })
        .collect()
}

/// 수치 기울기(행렬 입력): 행렬의 각 원소마다 편미분
/// 순전파(Loss) 함수를 받아 ∂Loss/∂W 를 행렬로 반환
fn numerical_gradient_matrix<F: Fn(&[Vec<f64>]) -> f64>(f: &F, w: &[Vec<f64>]) -> Matrix {
    let h = 1e-5;
    let mut grad = vec![vec![0.0; w[0].len()]; w.len()];
    for i in 0..w.len() {
        for j in 0..w[0].len() {
            let mut wp = w.to_vec();
            let mut wm = w.to_vec();
            wp[i][j] += h;
            wm[i][j] -= h;
            grad[i][j] = (f(&wp) - f(&wm)) / (2.0 * h);
        }
    }
    grad
}

/// 행렬 예쁘게 출력
fn print_matrix(m: &[Vec<f64>]) {
    for row in m {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:>7.2}")).collect();
        println!("  [{}]", cells.join(", "));
    }
    println!();
}