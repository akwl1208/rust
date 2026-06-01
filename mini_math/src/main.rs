fn main() {
    println!("================================================");
    println!(" Day 36-37: 미니 수학 라이브러리 (from scratch)");
    println!("================================================\n");
 
    part1_matrix();           // 행렬: 곱·전치
    part2_softmax();          // 소프트맥스
    part3_cross_entropy();    // 크로스 엔트로피
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

/// 행렬 예쁘게 출력
fn print_matrix(m: &[Vec<f64>]) {
    for row in m {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:>7.2}")).collect();
        println!("  [{}]", cells.join(", "));
    }
    println!();
}