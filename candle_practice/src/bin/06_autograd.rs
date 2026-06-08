// ================================================================
// candle 예제 06: 자동미분 (autograd) ★ 가장 중요
//
// 어떨 때 쓰나: 신경망 학습. Day 48-50에서 손으로 짠 역전파를
// candle이 '자동으로' 해준다. 이게 라이브러리를 쓰는 가장 큰 이유.
//
// 핵심 개념:
//   Var = '미분 추적되는 텐서'. 학습할 파라미터를 Var로 만든다.
//   연산을 하면 candle이 계산 그래프를 몰래 기록한다.
//   .backward()를 부르면 그래프를 거꾸로 따라가며 gradient를 자동 계산.
// ================================================================

use candle_core::{Tensor, Var, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 06: 자동미분 (autograd) ===\n");

    // --- 예제 1: 간단한 함수의 미분 ---
    // y = x^2 일 때 dy/dx = 2x. x=3이면 6이어야 한다.
    //
    // Var = 미분을 추적하는 텐서. (보통 텐서는 추적 안 함)
    let x = Var::new(3.0f32, &dev)?;

    // x를 텐서처럼 써서 연산. candle이 "x를 제곱했다"를 그래프에 기록.
    // (Var는 as_tensor()로 텐서처럼 다룬다)
    let y = x.as_tensor().sqr()?; // y = x^2

    // backward(): y에서 거꾸로 미분을 계산. 결과는 GradStore(기울기 모음).
    let grads = y.backward()?;

    // GradStore에서 x에 대한 gradient를 꺼낸다.
    let dx = grads.get(&x).unwrap();
    println!("y = x^2, x=3");
    println!("dy/dx (자동계산) = {}  (정답 2x=6)", dx.to_scalar::<f32>()?);
    println!("-> 우리가 손으로 미분 안 했는데 candle이 6을 구해줌!\n");

    // --- 예제 2: 여러 변수 ---
    // f = x^2 + 3y 일 때  df/dx = 2x,  df/dy = 3
    let x = Var::new(2.0f32, &dev)?;
    let y = Var::new(5.0f32, &dev)?;
    // f = x^2 + 3y
    let f = (x.as_tensor().sqr()? + y.as_tensor().affine(3.0, 0.0)?)?;
    let grads = f.backward()?;
    let dfdx = grads.get(&x).unwrap().to_scalar::<f32>()?;
    let dfdy = grads.get(&y).unwrap().to_scalar::<f32>()?;
    println!("f = x^2 + 3y  at (x=2, y=5)");
    println!("df/dx = {dfdx}  (정답 2x=4)");
    println!("df/dy = {dfdy}  (정답 3)");
    println!("-> 변수가 여러 개여도 각각의 gradient를 자동으로!\n");

    println!("PyTorch의 loss.backward()와 똑같은 원리.");
    Ok(())
}