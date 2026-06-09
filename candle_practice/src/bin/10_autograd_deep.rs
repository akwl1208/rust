// ================================================================
// candle 예제 10: 자동미분 심화 (Day 57-58) — 06의 후속
//
// 목표: "내가 손으로 했던 역전파를 .backward()가 똑같이 해준다"를 눈으로 확인.
//
// 핵심 키워드:
//   Var          : 미분을 추적하는 학습 파라미터 (gradient를 받는 대상)
//   .backward()  : 손실에서 거꾸로 기울기를 자동 계산
//   .get(&var)   : 특정 Var의 gradient 꺼내기 (PyTorch의 .grad)
//
// 실행:  cargo run --bin 10_autograd_deep
// ================================================================

use candle_core::{Tensor, Var, Device};
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== Day 57-58: candle autograd ===\n");
 
    // =============================================================
    // 파트 1: 가장 단순한 경우  loss = (w*x + b - y)^2
    //   손으로 미분하면:
    //     dL/dw = 2*(pred - y)*x
    //     dL/db = 2*(pred - y)
    //   이 값과 candle의 backward() 결과가 같은지 본다.
    // =============================================================
    println!("--- 파트 1: loss = (w*x + b - y)^2 ---");
 
    let x = Tensor::new(3.0f32, &dev)?;        // 입력 (상수)
    let y = Tensor::new(10.0f32, &dev)?;       // 정답 (상수)
    let w = Var::new(2.0f32, &dev)?;           // 학습 파라미터
    let b = Var::new(1.0f32, &dev)?;           // 학습 파라미터

    //순전파
    let pred = (w.as_tensor() * &x)?.add(b.as_tensor())?; // w*x + b = 7
    let err = (&pred - &y)?; // pred - y = -3
    let loss = err.sqr()?;
    println!("pred = {:.1}, loss = {:.1}", pred.to_scalar::<f32>()?, loss.to_scalar::<f32>()?);

    // 역전파: 이 한 줄이 dL/dw, dL/db 를 자동 계산한다.
    let grads = loss.backward()?;
    let dw = grads.get(&w).unwrap().to_scalar::<f32>()?;
    let db = grads.get(&b).unwrap().to_scalar::<f32>()?;

    // 손 계산값과 비교
    let pred_v = 2.0 * 3.0 + 1.0;      // 7
    let err_v = pred_v - 10.0;          // -3
    let dw_hand = 2.0 * err_v * 3.0;    // -18
    let db_hand = 2.0 * err_v;          // -6
    println!("candle backward(): dw={:.1}, db={:.1}", dw, db);
    println!("손 계산:           dw={:.1}, db={:.1}", dw_hand, db_hand);
    println!("=> 일치? {}\n", (dw - dw_hand).abs() < 1e-4 && (db - db_hand).abs() < 1e-4);

    // =============================================================
    // 파트 2: 2층 미니 신경망
    //   x -> z1 = w1*x + b1 -> h1 = relu(z1) -> out = w2*h1 + b2 -> loss
    //   레이어가 쌓여도 backward() 한 줄이면 모든 파라미터 기울기가 나온다.
    // =============================================================
    println!("--- 파트 2: 2층 신경망 (relu 포함) ---");
 
    let x = Tensor::new(1.0f32, &dev)?;
    let y = Tensor::new(2.0f32, &dev)?;
    let w1 = Var::new(0.5f32, &dev)?;
    let b1 = Var::new(0.0f32, &dev)?;
    let w2 = Var::new(0.8f32, &dev)?;
    let b2 = Var::new(0.1f32, &dev)?;
 
    // 순전파
    let z1 = (w1.as_tensor() * &x)?.add(b1.as_tensor())?; // 0.5
    let h1 = z1.relu()?;                                  // relu(0.5) = 0.5
    let out = (w2.as_tensor() * &h1)?.add(b2.as_tensor())?; // 0.8*0.5+0.1 = 0.5
    let loss = (&out - &y)?.sqr()?;                        // (0.5-2)^2 = 2.25
    println!("out = {:.2}, loss = {:.2}", out.to_scalar::<f32>()?, loss.to_scalar::<f32>()?);

    // 역전파 — 한 줄로 4개 파라미터 기울기 전부
    let grads = loss.backward()?;
    let g_w1 = grads.get(&w1).unwrap().to_scalar::<f32>()?;
    let g_b1 = grads.get(&b1).unwrap().to_scalar::<f32>()?;
    let g_w2 = grads.get(&w2).unwrap().to_scalar::<f32>()?;
    let g_b2 = grads.get(&b2).unwrap().to_scalar::<f32>()?;
    println!("candle: dw1={:.2}, db1={:.2}, dw2={:.2}, db2={:.2}", g_w1, g_b1, g_w2, g_b2);
    // 손 역전파 결과(파이썬으로 검증): dw1=-2.4, db1=-2.4, dw2=-1.5, db2=-3.0
    println!("손 계산: dw1=-2.40, db1=-2.40, dw2=-1.50, db2=-3.00");

    // =============================================================
    // 파트 3: no_grad — 추론할 땐 기울기 추적을 끈다
    //   학습이 끝나고 예측만 할 땐 gradient가 필요 없다.
    //   추적을 끄면 메모리/속도에서 이득.
    // =============================================================
    println!("\n--- 파트 3: no_grad (추론 모드) ---");
    let w = Var::new(2.0f32, &dev)?;
    let x = Tensor::new(3.0f32, &dev)?;
 
    // detach()로 그래프에서 떼어내면 그 텐서로는 backward가 흐르지 않는다.
    let pred_infer = (w.as_tensor().detach() * &x)?;
    println!("추론 결과 (grad 추적 안 함): {:.1}", pred_infer.to_scalar::<f32>()?);
    println!("=> 추론만 할 땐 detach로 그래프를 끊어 불필요한 추적을 피한다.");

    Ok(())
}