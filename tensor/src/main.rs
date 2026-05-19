fn main() {
    println!("========================================");
    println!(" Day 31-32: 벡터·행렬·텐서");
    println!("========================================\n");
 
    ex1_scalar_vector_matrix();
    ex2_mat_add_scalar_mul();
    ex3_matmul();

}

// ────────────────────────────────────────────
// 실습 1: 스칼라 · 벡터 · 행렬 · 텐서 개념
// ────────────────────────────────────────────
fn ex1_scalar_vector_matrix() {
    println!("── 실습 1: 스칼라 / 벡터 / 행렬 / 텐서 ──\n");

    //스칼라: 값 하나
    let scalar: f64 = 3.14;
    println!("스칼라: {scalar}");

    // 벡터: 1차원 배열 — Vec<f64>
    // shape = (4,)  rank = 1
    let vector: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    println!("벡터:   {:?}", vector);
    println!("  shape = ({},)  len = {}", vector.len(), vector.len());

    // 행렬: 2차원 — Vec<Vec<f64>>
    // shape = (2, 3)  rank = 2   
    let matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    let rows = matrix.len();
    let cols = matrix[0].len();
    println!("행렬:   {:?}", matrix);
    println!("  shape = ({rows}, {cols})  rank = 2");

    // 3D 텐서: LLM 입력 형태 시뮬레이션
    // shape = (batch=2, seq=3, dim=4)  rank = 3
    // (실제 LLM은 batch=4, seq=512, dim=768)
    let batch = 2;
    let seq = 3;
    let dim = 4;
    let tensor: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0; dim]; seq]; batch
    ];

    println!("3D 텐서 shape = ({batch}, {seq}, {dim})  rank = 3");
    println!("  원소 수 = {}  (실제 LLM: 4×512×768 = {})\n",
        batch * seq * dim,
        4 * 512 * 768);

    // 🔑 핵심 정리
    // ndim (rank) = 차원 수
    // shape       = 각 차원의 크기
    // size        = 전체 원소 수 (shape의 곱)        
}

// ────────────────────────────────────────────
// 실습 2: 행렬 덧셈 · 스칼라 곱
// ────────────────────────────────────────────
fn ex2_mat_add_scalar_mul() {
    println!("── 실습 2: 행렬 덧셈 / 스칼라 곱 ──\n");

    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],   
    ];
    let b = vec![
        vec![10.0, 20.0, 30.0],
        vec![40.0, 50.0, 60.0],
    ];

    // 원소별(element-wise) 덧셈
    let c = mat_add(&a, &b);
    println!("A + B (원소별 덧셈):");
    print_matrix(&c);

    // 스칼라 곱
    let d = scalar_mul(&a, 3.0);
    println!("A × 3 (스칼라 곱):");
    print_matrix(&d);

    // ⚠️ 주의: 원소별 곱(*)과 행렬 곱(matmul)은 완전히 다름!
    let e_elem = mat_elemwise_mul(&a, &b);
    println!("A * B (원소별 곱 — 행렬 곱 아님!):");
    print_matrix(&e_elem);
    println!();
}

// ────────────────────────────────────────────
// 실습 3: 행렬 곱 (가장 중요!)
// ────────────────────────────────────────────
fn ex3_matmul() {
    println!("── 실습 3: 행렬 곱 (matmul) ──\n");
 
    // A: (2×3)
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    // B: (3×2)
    let b = vec![
        vec![7.0,  8.0],
        vec![9.0,  10.0],
        vec![11.0, 12.0],
    ];

    println!("A shape: ({}, {})", a.len(), a[0].len());
    println!("B shape: ({}, {})", b.len(), b[0].len());
    println!("A:");
    print_matrix(&a);
    println!("B:");
    print_matrix(&b);

    let c = matmul(&a, &b);
 
    println!("C = A @ B:");
    print_matrix(&c);
    println!("C shape: ({}, {})  ← (2×3)·(3×2) = (2×2)", c.len(), c[0].len());

    // 손계산 과정 출력
    println!("\n손계산 과정:");
    println!("  C[0][0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = {}", 1*7 + 2*9 + 3*11);
    println!("  C[0][1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = {}", 1*8 + 2*10 + 3*12);
    println!("  C[1][0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = {}", 4*7 + 5*9 + 6*11);
    println!("  C[1][1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = {}", 4*8 + 5*10 + 6*12);

    // shape 규칙 검증
    println!("\n[shape 규칙] (m×k)·(k×n) = (m×n)");
    println!("  ({m}×{k})·({k}×{n}) = ({m}×{n}) ✓",
        m=2, k=3, n=2);

    // shape 불일치 시 어떻게 처리하는지
    let bad_b = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // (2×2) — 불일치
    println!("\n잘못된 shape ({m}×{k})·({k2}×{n}) 시도:",
        m=2, k=3, k2=2, n=2);
    match matmul_safe(&a, &bad_b) {
        Ok(r)  => print_matrix(&r),
        Err(e) => println!("  에러: {e}"),
    }
    println!();
}

// ================================================================
// 헬퍼 함수들 — 실제 구현부
// ================================================================
/// 행렬 출력
fn print_matrix(m: &Vec<Vec<f64>>) {
    for row in m {
        let formatted: Vec<String> = row.iter ()
            .map(|x| format!("{:7.2}", x))
            .collect();
        println!("  [{}]", formatted.join(", "));
    }
}

/// 행렬 덧셈 — A[i][j] + B[i][j]
fn mat_add(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    a.iter().zip(b.iter()) // 1. 두 행렬에서 줄(row)을 한 쌍씩 매칭
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b.iter()) // 2. 각 줄에서 숫자(x, y)를 한 쌍씩 매칭
                .map(|(x,y)| x + y) // 3. 같은 위치의 숫자끼리 더하기
                .collect()
        })
        .collect()
}

/// 원소별 곱 — A[i][j] * B[i][j]  (행렬 곱과 다름!)
fn mat_elemwise_mul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    a.iter().zip(b.iter())
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b.iter())
                .map(|(x,y)| x * y)
                .collect()
        })
        .collect()
}

/// 스칼라 곱 — A[i][j] * s
fn scalar_mul(a: &Vec<Vec<f64>>, s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|x| x * s).collect())
        .collect()
}

/// 행렬 곱 — C[i][j] = Σ A[i][k]*B[k][j]
/// A: (m×k)  B: (k×n)  →  C: (m×n)
fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();

    (0..m).map(|i| {
        (0..n).map(|j| {
            (0..k).map(|t| a[i][t] * b[t][j]).sum() 
        }).collect()
    }).collect()
}

/// shape 검증 포함 행렬 곱
fn matmul_safe(
    a: &Vec<Vec<f64>>,
    b: &Vec<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, String> {
    let k1 = a[0].len();
    let k2 = b.len();
    if k1 != k2 {
        return Err(format!(
            "shape 불일치: A의 열({k1}) ≠ B의 행({k2})"
        ));
    }
    Ok(matmul(a, b))
}