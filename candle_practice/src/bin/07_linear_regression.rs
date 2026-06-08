// ================================================================
// candle 예제 07: autograd로 선형회귀 학습 ★ 종합
//
// 어떨 때 쓰나: 실제 학습. Day 42-43에서 numpy로 손수 짠 선형회귀를,
// 이번엔 candle의 autograd로 짠다. 역전파를 직접 안 짜도 된다!
//
// y = 2x + 1 을 데이터로 주고, candle이 w≈2, b≈1 을 찾게 한다.
// ================================================================

use candle_core::{Tensor, Var, Device, D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 07: autograd 선형회귀 (y=2x+1 찾기) ===\n");

    // 데이터: x=[1,2,3,4,5], y=2x+1
    let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &dev)?;
    let y = Tensor::new(&[3.0f32, 5.0, 7.0, 9.0, 11.0], &dev)?;

    // 학습할 파라미터를 Var로 (미분 추적됨). 0에서 시작.
    let w = Var::new(0.0f32, &dev)?;
    let b = Var::new(0.0f32, &dev)?;
    let lr = 0.01;

    for step in 1..=200 {
        // --- 순전파: pred = w*x + b ---
        // w*x : 스칼라 Var를 텐서에 브로드캐스트 곱
        let pred = x.broadcast_mul(w.as_tensor())?
                    .broadcast_add(b.as_tensor())?;
        // --- 손실: MSE = mean((pred - y)^2) ---
        let loss = (pred - &y)?.sqr()?.mean(D::Minus1)?;

        // --- 역전파: candle이 자동으로! ---
        let grads = loss.backward()?;
        let dw = grads.get(&w).unwrap();
        let db = grads.get(&b).unwrap();

        // --- 업데이트: w = w - lr*dw  (Var는 set으로 값 갱신) ---
        let new_w = (w.as_tensor() - (dw * lr)?)?;
        let new_b = (b.as_tensor() - (db * lr)?)?;
        w.set(&new_w)?;
        b.set(&new_b)?;

        if step % 40 == 0 || step == 1 {
            println!("step {step:>3}: loss={:.5}  w={:.4}  b={:.4}",
                loss.to_scalar::<f32>()?,
                w.as_tensor().to_scalar::<f32>()?,
                b.as_tensor().to_scalar::<f32>()?);
        }
    }

    println!("\n최종: y = {:.4}x + {:.4}  (정답 y=2x+1)",
        w.as_tensor().to_scalar::<f32>()?,
        b.as_tensor().to_scalar::<f32>()?);
    println!("-> 역전파를 직접 안 짰는데 candle이 알아서 w,b를 찾았다!");

    Ok(())
}