const IN: usize = 3;   // 입력 차원
const HID: usize = 4;  // 은닉층 차원
const OUT: usize = 2;  // 출력 차원

// 파라미터 묶음 (W1: HIDxIN, b1: HID, W2: OUTxHID, b2: OUT)
#[derive(Clone)]
struct Params {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

fn init_params() -> Params {
    Params {
        w1: vec![
            vec![0.1, 0.2, -0.3],
            vec![0.4, -0.5, 0.6],
            vec![-0.7, 0.8, 0.1],
            vec![0.2, 0.3, -0.4],
        ],
        b1: vec![0.1, -0.2, 0.3, 0.0],
        w2: vec![
            vec![0.5, -0.6, 0.7, 0.1],
            vec![-0.2, 0.3, -0.4, 0.8],
        ],
        b2: vec![0.05, -0.05],
    }
}


fn main() {
    println!("====================================================");
    println!(" Day 48-50: 역전파 완전 이해 (Rust)");
    println!("====================================================");
    println!("신경망 [3->4->2], 은닉층 ReLU, 손실 MSE");
    println!("계산그래프: x -(W1)-> z1 -ReLU-> a1 -(W2)-> z2 -MSE-> loss\n");

    let x = vec![1.0, 2.0, -1.0];
    let target = vec![1.0, 0.0];

    // --- 1) Gradient Check ---
    println!("-- 1) Gradient Check (역전파 vs 수치미분) --\n");
    let params = init_params();
    let max_diff = gradient_check(&x, &target, &params);
    println!("해석적 기울기(역전파) vs 수치 미분 최대 오차: {max_diff:.2e}");
    let ok = if max_diff < 1e-5 { "성공! 역전파가 정확함" } else { "실패" };
    println!("1e-5 이내인가? -> {ok}");
    println!("(수치미분은 '느리지만 확실한' 정답. 역전파가 이와 같으면 OK)\n");

    // --- 2) 학습 루프 ---
    println!("-- 2) 학습 루프 (순전파->손실->역전파->업데이트) --\n");
    let mut params = init_params();
    let lr = 0.1;
    println!("{:>6} {:>12} {:>24}", "step", "loss", "output");
    println!("{}", "-".repeat(44));
    for step in 1..=200 {
        let (loss, out, cache) = forward(&x, &target, &params); // 순전파+손실
        let grads = backward(&target, &cache, &params);          // 역전파
        // 가중치 업데이트
        update(&mut params, &grads, lr);
        if [1, 10, 50, 100, 200].contains(&step) {
            println!("{step:>6} {loss:>12.6}   [{:>8.4}, {:>8.4}]", out[0], out[1]);
        }
    }
    println!("\n목표(target):           [{:>8.4}, {:>8.4}]", target[0], target[1]);
    println!("-> 역전파로 구한 기울기 방향으로 가중치를 옮기니 출력이 정답에 수렴!");
    println!("   이 '순전파->역전파->업데이트' 루프가 모든 신경망 학습의 핵심.");
    println!("   Autograd(candle/burn)는 backward()를 자동 생성해줄 뿐,");
    println!("   원리는 방금 손으로 짠 이 연쇄 법칙 그대로다.");
}


// ----------------------------------------------------------------
// 순전파 — 중간값(cache)을 저장해 역전파에서 재사용
// ----------------------------------------------------------------

struct Cache {
    x: Vec<f64>,
    z1: Vec<f64>,
    a1: Vec<f64>,
    z2: Vec<f64>,
}

fn forward(x: &[f64], target: &[f64], p: &Params) -> (f64, Vec<f64>, Cache) {
    // 은닉층: z1 = W1@x + b1, a1 = ReLU(z1)
    let mut z1 = vec![0.0; HID];
    let mut a1 = vec![0.0; HID];
    for i in 0..HID {
        let mut s = p.b1[i];
        for j in 0..IN {
            s += p.w1[i][j] * x[j];
        }
        z1[i] = s;
        a1[i] = if s > 0.0 { s } else { 0.0 }; // ReLU
    }
    // 출력층: z2 = W2@a1 + b2 (선형)
    let mut z2 = vec![0.0; OUT];
    for i in 0..OUT {
        let mut s = p.b2[i];
        for j in 0..HID {
            s += p.w2[i][j] * a1[j];
        }
        z2[i] = s;
    }
    // MSE 손실
    let mut loss = 0.0;
    for i in 0..OUT {
        loss += (z2[i] - target[i]).powi(2);
    }
    loss /= OUT as f64;

    let cache = Cache { x: x.to_vec(), z1, a1, z2: z2.clone() };
    (loss, z2, cache)
}


// ----------------------------------------------------------------
// 역전파 — 손으로 전개한 연쇄 법칙 그대로
// ----------------------------------------------------------------
// 1) L = mean((z2-t)^2)         -> dL/dz2 = 2(z2-t)/N
// 2) z2 = W2@a1 + b2            -> dW2 = dz2 (바깥곱) a1
//                                  db2 = dz2
//                                  da1 = W2^T @ dz2
// 3) a1 = ReLU(z1)             -> dz1 = da1 * (z1>0)
// 4) z1 = W1@x + b1            -> dW1 = dz1 (바깥곱) x
//                                  db1 = dz1

struct Grads {
    dw1: Vec<Vec<f64>>,
    db1: Vec<f64>,
    dw2: Vec<Vec<f64>>,
    db2: Vec<f64>,
}

fn backward(target: &[f64], c: &Cache, p: &Params) -> Grads {
    let n = OUT as f64;

    // 출력층: dz2 = 2(z2-t)/N
    let mut dz2 = vec![0.0; OUT];
    for i in 0..OUT {
        dz2[i] = 2.0 * (c.z2[i] - target[i]) / n;
    }
    // dW2 = dz2 (바깥곱) a1,  db2 = dz2
    let mut dw2 = vec![vec![0.0; HID]; OUT];
    let mut db2 = vec![0.0; OUT];
    for i in 0..OUT {
        db2[i] = dz2[i];
        for j in 0..HID {
            dw2[i][j] = dz2[i] * c.a1[j];
        }
    }
    // da1 = W2^T @ dz2
    let mut da1 = vec![0.0; HID];
    for j in 0..HID {
        let mut s = 0.0;
        for i in 0..OUT {
            s += p.w2[i][j] * dz2[i];
        }
        da1[j] = s;
    }
    // dz1 = da1 * ReLU'(z1)  (z1>0이면 1, 아니면 0)
    let mut dz1 = vec![0.0; HID];
    for j in 0..HID {
        dz1[j] = if c.z1[j] > 0.0 { da1[j] } else { 0.0 };
    }
    // dW1 = dz1 (바깥곱) x,  db1 = dz1
    let mut dw1 = vec![vec![0.0; IN]; HID];
    let mut db1 = vec![0.0; HID];
    for i in 0..HID {
        db1[i] = dz1[i];
        for j in 0..IN {
            dw1[i][j] = dz1[i] * c.x[j];
        }
    }

    Grads { dw1, db1, dw2, db2 }
}


// ----------------------------------------------------------------
// 가중치 업데이트 (경사하강법)
// ----------------------------------------------------------------

fn update(p: &mut Params, g: &Grads, lr: f64) {
    for i in 0..HID {
        p.b1[i] -= lr * g.db1[i];
        for j in 0..IN {
            p.w1[i][j] -= lr * g.dw1[i][j];
        }
    }
    for i in 0..OUT {
        p.b2[i] -= lr * g.db2[i];
        for j in 0..HID {
            p.w2[i][j] -= lr * g.dw2[i][j];
        }
    }
}


// ----------------------------------------------------------------
// Gradient Check — 역전파를 수치 미분으로 검증
// ----------------------------------------------------------------
// 수치 미분(중심차분): dL/dw ≈ [L(w+h) - L(w-h)] / 2h
// 모든 파라미터를 하나씩 살짝 흔들어 손실 변화를 보고,
// 역전파 기울기와 1e-5 이내로 같은지 확인한다.

fn gradient_check(x: &[f64], target: &[f64], p: &Params) -> f64 {
    let (_, _, cache) = forward(x, target, p);
    let g = backward(target, &cache, p);
    let h = 1e-5;
    let mut max_diff: f64 = 0.0;

    // W1 체크
    for i in 0..HID {
        for j in 0..IN {
            let mut pp = p.clone();
            pp.w1[i][j] += h;
            let (lp, _, _) = forward(x, target, &pp);
            pp.w1[i][j] -= 2.0 * h;
            let (lm, _, _) = forward(x, target, &pp);
            let num = (lp - lm) / (2.0 * h);
            max_diff = max_diff.max((num - g.dw1[i][j]).abs());
        }
    }
    // W2 체크
    for i in 0..OUT {
        for j in 0..HID {
            let mut pp = p.clone();
            pp.w2[i][j] += h;
            let (lp, _, _) = forward(x, target, &pp);
            pp.w2[i][j] -= 2.0 * h;
            let (lm, _, _) = forward(x, target, &pp);
            let num = (lp - lm) / (2.0 * h);
            max_diff = max_diff.max((num - g.dw2[i][j]).abs());
        }
    }
    max_diff
}