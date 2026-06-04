const N: usize = 4;    // 데이터 4개
const IN: usize = 2;   // 입력 차원
const HID: usize = 4;  // 은닉 차원

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn main() {
    println!("==================================================");
    println!(" Day 53-54: XOR 신경망 scratch [2->4->1] (Rust)");
    println!("==================================================");
    println!("XOR = 선형분리 불가능. 은닉층이 있어야 풀리는 고전 문제.\n");

    // XOR 데이터
    let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y = [0.0, 1.0, 1.0, 0.0];

    // 가중치 초기화 (재현 가능한 난수)
    let mut rng = Rng::new(42);
    let mut w1 = [[0.0; HID]; IN]; // 2x4
    let mut b1 = [0.0; HID];
    let mut w2 = [0.0; HID];       // 4x1
    let mut b2 = 0.0;
    for i in 0..IN {
        for j in 0..HID {
            w1[i][j] = rng.normal() * 0.5;
        }
    }
    for j in 0..HID {
        w2[j] = rng.normal() * 0.5;
    }

    let lr = 0.5;

    for ep in 1..=5000 {
        // 누적 기울기
        let mut dw1 = [[0.0; HID]; IN];
        let mut db1 = [0.0; HID];
        let mut dw2 = [0.0; HID];
        let mut db2 = 0.0;
        let mut loss = 0.0;

        for n in 0..N {
            // --- 순전파 ---
            let mut z1 = [0.0; HID];
            let mut a1 = [0.0; HID];
            for j in 0..HID {
                let mut s = b1[j];
                for i in 0..IN {
                    s += x[n][i] * w1[i][j];
                }
                z1[j] = s;
                a1[j] = sigmoid(s);
            }
            let mut z2 = b2;
            for j in 0..HID {
                z2 += a1[j] * w2[j];
            }
            let a2 = sigmoid(z2);
            loss += (a2 - y[n]).powi(2) / N as f64;

            // --- 역전파 ---
            // dL/dz2 = 2(a2-y)/N * a2(1-a2)
            let dz2 = 2.0 * (a2 - y[n]) / N as f64 * a2 * (1.0 - a2);
            db2 += dz2;
            for j in 0..HID {
                dw2[j] += dz2 * a1[j];
            }
            // 은닉층으로 전파
            for j in 0..HID {
                let da1 = dz2 * w2[j];
                let dz1 = da1 * a1[j] * (1.0 - a1[j]); // sigmoid 미분
                db1[j] += dz1;
                for i in 0..IN {
                    dw1[i][j] += dz1 * x[n][i];
                }
            }
        }

        // --- 업데이트 ---
        for i in 0..IN {
            for j in 0..HID {
                w1[i][j] -= lr * dw1[i][j];
            }
        }
        for j in 0..HID {
            b1[j] -= lr * db1[j];
            w2[j] -= lr * dw2[j];
        }
        b2 -= lr * db2;

        if [1, 1000, 3000, 5000].contains(&ep) {
            // 정확도 + 예측
            let mut preds = [0.0; N];
            let mut correct = 0;
            for n in 0..N {
                let mut a1 = [0.0; HID];
                for j in 0..HID {
                    let mut s = b1[j];
                    for i in 0..IN {
                        s += x[n][i] * w1[i][j];
                    }
                    a1[j] = sigmoid(s);
                }
                let mut z2 = b2;
                for j in 0..HID {
                    z2 += a1[j] * w2[j];
                }
                let a2 = sigmoid(z2);
                preds[n] = a2;
                if (a2 > 0.5) == (y[n] > 0.5) {
                    correct += 1;
                }
            }
            let acc = correct as f64 / N as f64;
            let pstr: Vec<String> = preds.iter().map(|p| format!("{p:.3}")).collect();
            println!("ep {ep:>4}: loss={loss:.5}  acc={acc:.2}  preds=[{}]", pstr.join(", "));
        }
    }

    println!("\n정답: {:?}", y);
    println!("-> 초반엔 0.5 근처에서 헤매다가 어느 순간 풀린다 (XOR의 전형).");
    println!("   numpy 없이 순수 Rust로 신경망 학습 성공!");
}

// 간단한 난수 생성기 (정규분포 포함)
struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_add(0x9E3779B97F4A7C15) } }
    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}