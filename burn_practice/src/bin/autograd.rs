
// ================================================================
// Day 57-58: Autograd in Rust — burn 편
//
// 목표: burn의 AutodiffBackend 트레이트로 자동미분을 켜는 법을 익힌다.
//
// candle과 가장 큰 차이:
//   candle : Var로 파라미터를 만들면 자동 추적
//   burn   : 백엔드를 Autodiff<...>로 '감싸야' 추적 가능
//            + 추적할 텐서에 .require_grad() 를 붙인다
//            + 기울기는 x.grad(&grads) 로 꺼낸다
//
// Cargo.toml:
//   [dependencies]
//   burn = { version = "0.21", features = ["ndarray", "autodiff"] }
//
// 실행:  cargo run --bin autograd
// ================================================================
 
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
 
// 핵심! NdArray(CPU)를 Autodiff로 감싼다.
//   이렇게 감싼 백엔드만 .backward() 를 쓸 수 있다.
//   감싸지 않은 NdArray로는 애초에 backward 호출이 불가능 -> 추론 전용.
type B = Autodiff<NdArray>;
 
fn main() {
    let device = Default::default();
    println!("=== Day 57-58: burn autograd ===\n");
 
    // =============================================================
    // 파트 1: 단순 곱셈+덧셈  y = a * b + c 의 기울기
    //   손 미분: dy/da = b,  dy/db = a  (c는 require_grad 안 했으므로 기울기 없음)
    // =============================================================
    println!("--- 파트 1: y = a*b + c ---");
 
    // require_grad() = "이 텐서의 기울기를 추적해줘" 표시 (candle의 Var 역할)
    let a = Tensor::<B, 1>::from_data([10.0f32], &device).require_grad();
    let b = Tensor::<B, 1>::from_data([5.0f32], &device).require_grad();
    let c = Tensor::<B, 1>::from_data([2.0f32], &device); // 추적 안 함 (상수 취급)

    // 순전파. burn은 값을 소비하므로 뒤에서 또 쓸 거면 clone.
    let y = a.clone() * b.clone() + c; // 10*5 + 2 = 52

    // 역전파 — 모든 require_grad 텐서의 기울기를 한 번에 계산
    let grads = y.backward();

    // 기울기 꺼내기: x.grad(&grads)
    let grad_a = a.grad(&grads).unwrap();
    let grad_b = b.grad(&grads).unwrap();
    println!("y = {:?}", y.into_data().to_vec::<f32>().unwrap());
    println!("burn: dy/da = {:?}  (손 계산: b = 5)", grad_a.into_data().to_vec::<f32>().unwrap());
    println!("burn: dy/db = {:?}  (손 계산: a = 10)", grad_b.into_data().to_vec::<f32>().unwrap());

    // =============================================================
    // 파트 2: 선형 레이어 하나  loss = (w*x + b - y)^2
    //   candle 편과 같은 식. 손 미분:
    //     dL/dw = 2*(pred-y)*x,  dL/db = 2*(pred-y)
    //   x=3, y=10, w=2, b=1 이면 pred=7, err=-3 -> dw=-18, db=-6
    // =============================================================
    println!("\n--- 파트 2: loss = (w*x + b - y)^2 ---");
 
    let x = Tensor::<B, 1>::from_data([3.0f32], &device);          // 입력 (상수)
    let target = Tensor::<B, 1>::from_data([10.0f32], &device);    // 정답 (상수)
    let w = Tensor::<B, 1>::from_data([2.0f32], &device).require_grad();
    let bias = Tensor::<B, 1>::from_data([1.0f32], &device).require_grad();

     let pred = w.clone() * x + bias.clone();      // w*x + b = 7
    let err = pred - target;                       // -3
    let loss = err.clone() * err;                  // (-3)^2 = 9  (제곱 = err*err)
 
    let grads = loss.backward();
    let gw = w.grad(&grads).unwrap();
    let gb = bias.grad(&grads).unwrap();
    println!("loss = {:?}", loss.into_data().to_vec::<f32>().unwrap());
    println!("burn: dw = {:?}  (손 계산: -18)", gw.into_data().to_vec::<f32>().unwrap());
    println!("burn: db = {:?}  (손 계산: -6)",  gb.into_data().to_vec::<f32>().unwrap());

    // =============================================================
    // 파트 3: 추론(no_grad)은 burn에선 타입으로 강제된다
    //   Autodiff로 안 감싼 백엔드(NdArray)는 backward 메서드 자체가 없다.
    //   => "추론용 백엔드로 실수로 backward 부르기"가 컴파일 단계에서 차단됨.
    // =============================================================
    println!("\n--- 파트 3: 추론 모드 (no_grad) ---");
    println!("burn에선 추론용 백엔드(Autodiff로 안 감싼 것)엔 backward()가 없다.");
    println!("=> '추론인데 실수로 역전파' 같은 버그를 컴파일러가 막아준다.");
    println!("   특정 텐서만 끄려면 .detach() 사용.");
}
