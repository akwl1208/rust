// ================================================================
// Day 44: 옵티마이저 심화 — SGD vs Momentum vs Adam vs AdamW (Rust)
//
// "왜 LLM 파인튜닝은 AdamW를 표준으로 쓰는가?"를 코드로 체감합니다.
//
// 발전 흐름:
//   SGD       -> 모든 파라미터에 같은 학습률 (단순하지만 느리고 진동)
//   Momentum  -> 이전 방향을 기억해 진동을 줄이고 가속
//   Adam      -> 파라미터마다 학습률을 자동 조정 (Momentum + 적응형)
//   AdamW     -> Adam + Weight Decay (과적합 방지) = LLM 파인튜닝 표준
//
// 외부 크레이트 없음. cargo run 으로 실행하세요.
// ================================================================

// ----------------------------------------------------------------
// 테스트용 손실함수: f(x, y) = x^2 + 10*y^2  (비대칭 계곡)
// ----------------------------------------------------------------
// y 방향 경사가 매우 가파르고(계수 10), x 방향은 완만(계수 1).
// 이런 지형에서 옵티마이저 차이가 극명하게 드러난다.
// 최저점은 (0,0), 그때 손실 = 0.

fn loss(p: [f64; 2]) -> f64 {
    p[0] * p[0] + 10.0 * p[1] * p[1]
}

fn grad(p: [f64; 2]) -> [f64; 2] {
    [2.0 * p[0], 20.0 * p[1]] // [df/dx, df/dy]
}

const START: [f64; 2] = [5.0, 5.0]; // 모두 같은 출발점
const STEPS: usize = 60;


fn main() {
    println!("========================================================");
    println!(" Day 44: 옵티마이저 심화 — Adam & AdamW (Rust)");
    println!("========================================================");
    println!("손실함수 f(x,y) = x^2 + 10y^2, 출발점 (5,5), 최저점 (0,0)\n");

    let h_sgd = sgd(0.01);
    let h_mom = momentum(0.01, 0.9);
    let h_adam = adam(0.3, 0.9, 0.999, 1e-8);

    // 실험 A: 수렴 속도 비교
    println!("-- 실험 A: 수렴 속도 비교 --\n");
    println!("{:>6} {:>14} {:>14} {:>14}", "step", "SGD", "Momentum", "Adam");
    println!("{}", "-".repeat(50));
    for s in [0, 5, 10, 20, 40, 60] {
        println!("{s:>6} {:>14.6} {:>14.6} {:>14.6}", h_sgd[s], h_mom[s], h_adam[s]);
    }
    println!();
    println!("최종 손실:  SGD={:.6}  Momentum={:.6}  Adam={:.6}",
        h_sgd[STEPS], h_mom[STEPS], h_adam[STEPS]);
    println!("-> SGD는 느리게, Momentum은 진동하며 가속, Adam은 적응형 수렴\n");

    // ASCII 수렴 곡선
    plot_ascii("SGD", &h_sgd);
    plot_ascii("Momentum", &h_mom);
    plot_ascii("Adam", &h_adam);

    // 실험 B: AdamW weight decay 효과
    println!("\n-- 실험 B: AdamW의 weight decay 효과 --\n");
    println!("{:>8} {:>14} {:>16}", "wd", "최종 손실", "최종 |params|");
    println!("{}", "-".repeat(40));
    for wd in [0.0, 0.01, 0.1] {
        let (final_loss, final_norm) = adamw(0.3, wd, 0.9, 0.999, 1e-8);
        println!("{wd:>8} {final_loss:>14.6} {final_norm:>16.5}");
    }
    println!();
    println!("-> wd가 클수록 가중치가 0쪽으로 더 당겨진다(크기 작아짐).");
    println!("   가중치가 작으면 모델이 단순해져 과적합이 줄어든다.\n");

    // 정리
    println!("========================================================");
    println!(" 정리");
    println!("========================================================");
    println!("SGD      : 같은 학습률 -> 느리고 진동");
    println!("Momentum : 관성으로 진동 억제 + 가속");
    println!("Adam     : 파라미터별 학습률 자동 조정 (b1=0.9, b2=0.999)");
    println!("AdamW    : Adam + 분리된 weight decay -> LLM 파인튜닝 표준");
    println!("\n튜닝 포인트: learning rate(보폭), weight decay(과적합 억제 강도)");
}


// ----------------------------------------------------------------
// 1. SGD
// ----------------------------------------------------------------
// 기울기 반대 방향으로 일정 보폭만큼:  p <- p - lr * grad
// 문제: 모든 파라미터에 '같은' 학습률 -> 가파른 방향은 진동, 완만한 방향은 느림.

fn sgd(lr: f64) -> Vec<f64> {
    let mut p = START;
    let mut history = vec![loss(p)];
    for _ in 0..STEPS {
        let g = grad(p);
        p[0] -= lr * g[0];
        p[1] -= lr * g[1];
        history.push(loss(p));
    }
    history
}


// ----------------------------------------------------------------
// 2. Momentum
// ----------------------------------------------------------------
// '관성' 도입. 이전 속도 v를 기억해 누적:
//   v <- beta*v + grad
//   p <- p - lr*v
// 같은 방향이면 가속, 좌우 진동은 상쇄되어 줄어든다.

fn momentum(lr: f64, beta: f64) -> Vec<f64> {
    let mut p = START;
    let mut v = [0.0, 0.0];
    let mut history = vec![loss(p)];
    for _ in 0..STEPS {
        let g = grad(p);
        for i in 0..2 {
            v[i] = beta * v[i] + g[i];
            p[i] -= lr * v[i];
        }
        history.push(loss(p));
    }
    history
}


// ----------------------------------------------------------------
// 3. Adam (Adaptive Moment Estimation)
// ----------------------------------------------------------------
//   m: 기울기 평균 (1차 모멘트) = 방향 (Momentum 역할)
//   v: 기울기 제곱 평균 (2차 모멘트) = 그 파라미터의 출렁임 정도
//   m <- b1*m + (1-b1)*g
//   v <- b2*v + (1-b2)*g^2
//   편향보정: m_hat = m/(1-b1^t), v_hat = v/(1-b2^t)
//   p <- p - lr * m_hat / (sqrt(v_hat)+eps)
// 많이 출렁인 파라미터는 학습률을 자동으로 줄인다 -> 파라미터별 적응.

fn adam(lr: f64, b1: f64, b2: f64, eps: f64) -> Vec<f64> {
    let mut p = START;
    let mut m = [0.0, 0.0];
    let mut v = [0.0, 0.0];
    let mut history = vec![loss(p)];
    for t in 1..=STEPS {
        let g = grad(p);
        for i in 0..2 {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - b1.powi(t as i32)); // 편향 보정
            let v_hat = v[i] / (1.0 - b2.powi(t as i32));
            p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        history.push(loss(p));
    }
    history
}


// ----------------------------------------------------------------
// 4. AdamW (Adam + 분리된 Weight Decay) — LLM 파인튜닝 표준
// ----------------------------------------------------------------
// weight decay = 가중치를 매 스텝 조금씩 0쪽으로 끌어당김(과적합 방지).
//
// AdamW의 핵심: decay를 gradient에 섞지 않고 '분리'해서 직접 적용:
//   p <- p - lr*(m_hat/(sqrt(v_hat)+eps))  -  lr*wd*p
//            └─ Adam 업데이트 ─┘             └─ 분리된 decay ─┘
// 이 'decoupled weight decay'가 AdamW의 W.
//
// 반환: (최종 손실, 최종 가중치 크기)

fn adamw(lr: f64, wd: f64, b1: f64, b2: f64, eps: f64) -> (f64, f64) {
    let mut p = START;
    let mut m = [0.0, 0.0];
    let mut v = [0.0, 0.0];
    for t in 1..=STEPS {
        let g = grad(p);
        for i in 0..2 {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - b1.powi(t as i32));
            let v_hat = v[i] / (1.0 - b2.powi(t as i32));
            // Adam 업데이트 + 분리된 weight decay
            p[i] -= lr * m_hat / (v_hat.sqrt() + eps) + lr * wd * p[i];
        }
    }
    let norm = (p[0] * p[0] + p[1] * p[1]).sqrt();
    (loss(p), norm)
}


// ----------------------------------------------------------------
// ASCII 수렴 곡선 (matplotlib 대체)
// ----------------------------------------------------------------

fn plot_ascii(name: &str, history: &[f64]) {
    println!("[{name}] 손실 감소 (대표 step):");
    let max_loss = history.iter().cloned().fold(f64::MIN, f64::max);
    let width = 40;
    for (i, &l) in history.iter().enumerate() {
        if i == 0 || i % 10 == 0 {
            let filled = ((l / max_loss) * width as f64).round() as usize;
            let bar: String = "#".repeat(filled);
            println!("  ep{i:>3} |{bar:<width$}| {l:.4}", width = width);
        }
    }
    println!();
}