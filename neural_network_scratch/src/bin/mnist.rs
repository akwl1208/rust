// ================================================================
// Day 53-54: 신경망 scratch 구현 — 숫자 분류 (Rust)
//
// numpy 없이 순수 Rust로 [64 -> 32 -> 10] 다중분류 신경망.
// ReLU + softmax + cross-entropy, 미니배치 전체(batch GD).
//
// 진짜 MNIST는 다운로드가 필요하므로, 여기서는 재현 가능한
// '합성 숫자 패턴'을 생성한다. 클래스(0~9)마다 고유한 픽셀 패턴에
// 노이즈를 더해 만들며, 분류 원리/구조는 MNIST와 완전히 동일하다.
//   입력 64(=8x8) -> 은닉 32 (ReLU) -> 출력 10 (softmax)
//
// 외부 크레이트 없음. cargo run 으로 실행하세요.
// ================================================================

const PIX: usize = 64;    // 8x8 픽셀
const HID: usize = 32;    // 은닉 차원
const CLASSES: usize = 10;
const N_TRAIN: usize = 500;
const N_TEST: usize = 100;

fn relu(z: f64) -> f64 { if z > 0.0 { z } else { 0.0 } }

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// 클래스 c의 '원형(prototype)' 패턴 생성 (클래스마다 다른 픽셀이 켜짐)
fn prototype(c: usize) -> [f64; PIX] {
    let mut p = [0.0; PIX];
    // 클래스마다 다른 위치의 픽셀 묶음을 켠다 (구분 가능한 고유 패턴)
    for k in 0..PIX {
        // c에 따라 다른 주기/위상으로 패턴 형성
        let v = ((k * (c + 1)) % 7) as f64 / 7.0
              + (((k + c) % 5) as f64 / 5.0) * 0.5;
        p[k] = v;
    }
    p
}

// 한 샘플 생성: 클래스 c의 원형 + 노이즈
fn make_sample(c: usize, rng: &mut Rng) -> [f64; PIX] {
    let proto = prototype(c);
    let mut s = [0.0; PIX];
    for k in 0..PIX {
        s[k] = (proto[k] + rng.normal() * 0.15).clamp(0.0, 1.0);
    }
    s
}

fn main() {
    println!("==================================================");
    println!(" Day 53-54: 숫자 분류 신경망 scratch [64->32->10]");
    println!("==================================================");
    println!("합성 숫자 패턴(8x8), ReLU + softmax + cross-entropy");
    println!("(진짜 MNIST와 구조/원리 동일, 다운로드 없이 학습)\n");

    let mut rng = Rng::new(7);

    // 데이터 생성 (학습 500 + 테스트 100)
    let mut xtr = vec![[0.0; PIX]; N_TRAIN];
    let mut ytr = vec![0usize; N_TRAIN];
    for i in 0..N_TRAIN {
        let c = i % CLASSES;
        xtr[i] = make_sample(c, &mut rng);
        ytr[i] = c;
    }
    let mut xte = vec![[0.0; PIX]; N_TEST];
    let mut yte = vec![0usize; N_TEST];
    for i in 0..N_TEST {
        let c = i % CLASSES;
        xte[i] = make_sample(c, &mut rng);
        yte[i] = c;
    }

    // 가중치 (He 초기화)
    let mut w1 = vec![[0.0; HID]; PIX]; // 64x32
    let mut b1 = [0.0; HID];
    let mut w2 = vec![[0.0; CLASSES]; HID]; // 32x10
    let mut b2 = [0.0; CLASSES];
    let s1 = (2.0 / PIX as f64).sqrt();
    let s2 = (2.0 / HID as f64).sqrt();
    for i in 0..PIX { for j in 0..HID { w1[i][j] = rng.normal() * s1; } }
    for i in 0..HID { for j in 0..CLASSES { w2[i][j] = rng.normal() * s2; } }

    let lr = 0.5;

    for ep in 1..=100 {
        // 누적 기울기
        let mut dw1 = vec![[0.0; HID]; PIX];
        let mut db1 = [0.0; HID];
        let mut dw2 = vec![[0.0; CLASSES]; HID];
        let mut db2 = [0.0; CLASSES];
        let mut loss = 0.0;

        for n in 0..N_TRAIN {
            // --- 순전파 ---
            let mut z1 = [0.0; HID];
            let mut a1 = [0.0; HID];
            for j in 0..HID {
                let mut s = b1[j];
                for i in 0..PIX { s += xtr[n][i] * w1[i][j]; }
                z1[j] = s;
                a1[j] = relu(s);
            }
            let mut z2 = [0.0; CLASSES];
            for j in 0..CLASSES {
                let mut s = b2[j];
                for i in 0..HID { s += a1[i] * w2[i][j]; }
                z2[j] = s;
            }
            let p = softmax(&z2);
            loss += -(p[ytr[n]] + 1e-9).ln() / N_TRAIN as f64;

            // --- 역전파 ---
            // softmax+CE: dz2 = (p - onehot)/N
            let mut dz2 = [0.0; CLASSES];
            for j in 0..CLASSES {
                let target = if j == ytr[n] { 1.0 } else { 0.0 };
                dz2[j] = (p[j] - target) / N_TRAIN as f64;
            }
            for j in 0..CLASSES {
                db2[j] += dz2[j];
                for i in 0..HID { dw2[i][j] += dz2[j] * a1[i]; }
            }
            // 은닉층 전파
            for i in 0..HID {
                let mut da1 = 0.0;
                for j in 0..CLASSES { da1 += dz2[j] * w2[i][j]; }
                let dz1 = if z1[i] > 0.0 { da1 } else { 0.0 }; // ReLU 미분
                db1[i] += dz1;
                for k in 0..PIX { dw1[k][i] += dz1 * xtr[n][k]; }
            }
        }

        // --- 업데이트 ---
        for i in 0..PIX { for j in 0..HID { w1[i][j] -= lr * dw1[i][j]; } }
        for j in 0..HID { b1[j] -= lr * db1[j]; }
        for i in 0..HID { for j in 0..CLASSES { w2[i][j] -= lr * dw2[i][j]; } }
        for j in 0..CLASSES { b2[j] -= lr * db2[j]; }

        if [1, 10, 30, 50, 100].contains(&ep) {
            let acc = test_accuracy(&xte, &yte, &w1, &b1, &w2, &b2);
            println!("ep {ep:>3}: loss={loss:.4}  test_acc={acc:.4}");
        }
    }

    let acc = test_accuracy(&xte, &yte, &w1, &b1, &w2, &b2);
    println!("\n최종 테스트 정확도: {acc:.4}  (무작위 추측이면 0.10)");
    println!("-> numpy 없이 순수 Rust로 10-클래스 분류 신경망 학습 성공!");
}

fn test_accuracy(
    xte: &[[f64; PIX]], yte: &[usize],
    w1: &[[f64; HID]], b1: &[f64; HID],
    w2: &[[f64; CLASSES]], b2: &[f64; CLASSES],
) -> f64 {
    let mut correct = 0;
    for n in 0..xte.len() {
        let mut a1 = [0.0; HID];
        for j in 0..HID {
            let mut s = b1[j];
            for i in 0..PIX { s += xte[n][i] * w1[i][j]; }
            a1[j] = relu(s);
        }
        let mut z2 = [0.0; CLASSES];
        for j in 0..CLASSES {
            let mut s = b2[j];
            for i in 0..HID { s += a1[i] * w2[i][j]; }
            z2[j] = s;
        }
        // argmax
        let mut best = 0;
        for j in 1..CLASSES { if z2[j] > z2[best] { best = j; } }
        if best == yte[n] { correct += 1; }
    }
    correct as f64 / xte.len() as f64
}

// 난수 생성기
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