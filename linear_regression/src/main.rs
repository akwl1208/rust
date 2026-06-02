fn main() {
    println!("==================================================");
    println!(" Day 42-43: Rust 선형회귀 완전 구현");
    println!("==================================================");
    println!("진짜 관계: y = 2x + 1 (+ 노이즈)");
    println!("목표: 데이터만 보고 w=2, b=1 을 스스로 찾기\n");

    // 1) 데이터 생성
    let (xs, ys) = make_data(100, 2.0, 1.0, 1.5);

    // 2) 입력 정규화 (학습은 정규화된 공간에서)
    let (xn, x_mean, x_std) = normalize(&xs);

    // 3) 학습
    let (w_n, b_n, loss_history) = train(&xn, &ys, 0.1, 100);

    // 4) 정규화 공간의 (w,b)를 원래 x 공간으로 환산
    //    y = w_n*((x-mean)/std) + b_n = (w_n/std)*x + (b_n - w_n*mean/std)
    let w = w_n / x_std;
    let b = b_n - w_n * x_mean / x_std;

    println!("\n학습 완료 (원래 공간): y = {w:.4}x + {b:.4}");
    println!("(노이즈 때문에 정확히 2,1은 아니지만 매우 근접)\n");
 
    // 5) ASCII 학습 곡선 (matplotlib 대체)
    plot_ascii_loss(&loss_history);
 
    // 6) 정규방정식으로 검증 (sklearn 대체)
    compare_normal_equation(&xs, &ys, w, b);
}

// ----------------------------------------------------------------
// 1. 데이터 생성
// ----------------------------------------------------------------
// 진짜 관계 y = true_w*x + true_b 에 정규분포 노이즈를 더한다.
// 외부 크레이트 없이 난수를 만들기 위해 간단한 난수 생성기를 직접 구현.
 
fn make_data(n: usize, true_w: f64, true_b: f64, noise: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(42);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.uniform(0.0, 10.0); //입력 0~10
        let y = true_w * x + true_b + rng.normal() * noise; // 정답 + 노이즈
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

// ----------------------------------------------------------------
// 2. 입력 정규화 (표준화)
// ----------------------------------------------------------------
// x' = (x - 평균) / 표준편차  -> 평균0, 표준편차1
//
// 왜? x의 평균이 5처럼 0에서 멀면 절편 b의 수렴이 매우 느려진다.
// (w는 금방 찾는데 b만 한참 뒤처짐) 입력을 0 중심으로 맞추면
// w, b가 비슷한 속도로 함께 수렴한다. -> 실무의 표준 전처리.
 
fn normalize(xs: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let normalized = xs.iter().map(|x| (x - mean) / std).collect();
    (normalized, mean, std)
}

// ----------------------------------------------------------------
// 3. 모델 / 손실 / 기울기
// ----------------------------------------------------------------
// 모델:  y_hat = w*x + b
// 손실:  MSE = (1/N) sum (y_hat - y)^2
// 기울기:
//   dL/dw = (2/N) sum (y_hat - y) * x
//   dL/db = (2/N) sum (y_hat - y)
 
fn mse_loss(xs: &[f64], ys: &[f64], w: f64, b: f64) -> f64 {
    let n = xs.len() as f64;
    xs.iter()
        .zip(ys)
        .map(|(x, y)| (w * x + b - y).powi(2))
        .sum::<f64>()
        / n
}

fn gradients(xs: &[f64], ys: &[f64], w: f64, b: f64) -> (f64, f64) {
    let n = xs.len() as f64;
    let mut dw = 0.0;
    let mut db = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let err = w * x + b - y; // 순전파 후 오차
        dw += err * x;
        db += err;
    }
    (2.0 / n * dw, 2.0 / n * db)
}

// ----------------------------------------------------------------
// 4. 학습 루프 (핵심 - LLM 파인튜닝과 구조 동일)
// ----------------------------------------------------------------
 
fn train(xs: &[f64], ys: &[f64], lr: f64, epochs: usize) -> (f64, f64, Vec<f64>) {
    let mut w = 0.0; // 모델 초기화: 백지 상태
    let mut b = 0.0;
    let mut loss_history = Vec::with_capacity(epochs);
 
    println!("{:>6} {:>14} {:>10} {:>10}", "epoch", "loss", "w", "b");
    println!("{}", "-".repeat(44));
 
    for epoch in 1..=epochs {
        let loss = mse_loss(xs, ys, w, b);       // 2) 손실 (순전파 포함)
        let (dw, db) = gradients(xs, ys, w, b);  // 3) 역전파(기울기)
        w -= lr * dw;                            // 4) 업데이트
        b -= lr * db;
 
        loss_history.push(loss);
 
        if epoch % 10 == 0 || epoch == 1 {
            println!("{epoch:>6} {loss:>14.6} {w:>10.4} {b:>10.4}");
        }
    }
    (w, b, loss_history)
}

// ----------------------------------------------------------------
// 5. ASCII 학습 곡선 (matplotlib 대체)
// ----------------------------------------------------------------
// loss가 epoch에 따라 줄어드는 모습을 터미널에 막대로 그린다.
 
fn plot_ascii_loss(loss_history: &[f64]) {
    println!("학습 곡선 (loss가 줄어드는 모습):");
    println!("{}", "-".repeat(60));
 
    let max_loss = loss_history.iter().cloned().fold(f64::MIN, f64::max);
    let bar_width = 45;
 
    // 대표 지점만 출력 (1, 10, 20, ..., 100)
    for (i, &loss) in loss_history.iter().enumerate() {
        let epoch = i + 1;
        if epoch == 1 || epoch % 10 == 0 {
            let filled = ((loss / max_loss) * bar_width as f64).round() as usize;
            let bar: String = "#".repeat(filled);
            println!("ep{epoch:>3} |{bar:<width$}| {loss:.4}", width = bar_width);
        }
    }
    println!("{}", "-".repeat(60));
    println!("-> 막대가 빠르게 짧아짐 = loss가 급격히 감소 = 학습 성공\n");
}
 
 
// ----------------------------------------------------------------
// 6. 정규방정식으로 검증 (sklearn 대체)
// ----------------------------------------------------------------
// 단순 선형회귀의 닫힌 해(closed-form):
//   w = sum((x-x̄)(y-ȳ)) / sum((x-x̄)^2)
//   b = ȳ - w*x̄
// 이게 수학적으로 '정확한' 최적해. 경사하강법 결과와 비교.
 
fn compare_normal_equation(xs: &[f64], ys: &[f64], my_w: f64, my_b: f64) {
    let n = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;
 
    let mut num = 0.0; // 분자
    let mut den = 0.0; // 분모
    for (x, y) in xs.iter().zip(ys) {
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean).powi(2);
    }
    let ne_w = num / den;
    let ne_b = y_mean - ne_w * x_mean;
 
    println!("==================================================");
    println!(" 정규방정식 비교 검증 (sklearn 대체)");
    println!("==================================================");
    println!("{:>16}{:>14}{:>14}", "", "w (기울기)", "b (절편)");
    println!("{}", "-".repeat(44));
    println!("{:>16}{:>14.4}{:>14.4}", "내 경사하강법", my_w, my_b);
    println!("{:>16}{:>14.4}{:>14.4}", "정규방정식(정답)", ne_w, ne_b);
    println!("{:>16}{:>14.4}{:>14.4}", "차이", (my_w - ne_w).abs(), (my_b - ne_b).abs());
    println!("{}", "-".repeat(44));
    if (my_w - ne_w).abs() < 0.05 && (my_b - ne_b).abs() < 0.05 {
        println!("=> 거의 완벽히 일치! 직접 만든 경사하강법이 정확히 동작함");
    } else {
        println!("=> 약간의 차이. epoch나 학습률을 조정해볼 수 있음");
    }
    println!("(정규방정식은 한 번에 '정확한' 해를 계산,");
    println!(" 경사하강법은 '점진적으로' 도달 -> 둘이 일치하면 성공)");
}
 
 
// ----------------------------------------------------------------
// 부록: 간단한 난수 생성기 (외부 크레이트 없이 재현 가능한 난수)
// ----------------------------------------------------------------
// LCG(선형 합동 생성기)로 균등분포를, Box-Muller로 정규분포를 만든다.
// 학습용이라 품질보다 '의존성 없이 재현 가능' 에 초점.
 
struct Rng {
    state: u64,
}
 
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }
 
    // 다음 난수 (0~1 사이 f64)
    fn next_f64(&mut self) -> f64 {
        // glibc 계열 LCG 상수
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // 상위 53비트를 [0,1)로
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
 
    // 균등분포 [lo, hi)
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
 
    // 표준정규분포 N(0,1) - Box-Muller 변환
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12); // log(0) 방지
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}