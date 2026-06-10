// Day 59-60: burn으로 MLP 완전 구현 (XOR 분류)
//
// 목표: Week 7에서 numpy로 만든 2층 신경망을 burn으로 재구현.
//   XOR = 비선형 문제. 직선 하나로는 못 가르고, 은닉층(relu)이 있어야 풀린다.
//   => relu를 왜 배웠는지 여기서 실감하게 된다.
//
// 핵심 키워드:
//   #[derive(Module)] : 이 struct가 신경망 '모듈'임을 표시. 파라미터 자동 관리.
//   #[derive(Config)] : 하이퍼파라미터 묶음을 만드는 매크로.
//   AdamConfig        : Adam 옵티마이저 설정 (경사하강법의 똑똑한 버전).
//   forward→loss→backward→step : 학습 루프 4단계.
//
// Cargo.toml:
//   [dependencies]
//   burn = { version = "0.21", features = ["ndarray", "autodiff", "train"] }
//
// 실행:  cargo run -- bin mlp_xor
// ================================================================
 
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::nn::loss::MseLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use burn::backend::{Autodiff, NdArray};
 
// 학습엔 자동미분이 필요하므로 NdArray(CPU)를 Autodiff로 감싼다.
type MyBackend = Autodiff<NdArray>;

// ================================================================
// 1) 모델 정의
//   #[derive(Module)]를 붙이면 burn이 이 struct를 신경망으로 인식하고
//   안에 든 Linear들의 파라미터(가중치·편향)를 자동으로 추적/업데이트한다.
//   구조: 입력 2 -> 은닉 8 (relu) -> 출력 1
// ================================================================
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linear1: Linear<B>, // 2 -> 8
    relu: Relu,         // 비선형 (이게 없으면 1층이랑 똑같아짐)
    linear2: Linear<B>, // 8 -> 1
}
 
impl<B: Backend> Mlp<B> {
    // 모델 초기화: 레이어 크기를 정해서 만든다.
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(2, 8).init(device),
            relu: Relu::new(),
            linear2: LinearConfig::new(8, 1).init(device),
        }
    }
 
    // 순전파: 입력 -> 은닉(relu) -> 출력
    //   numpy 때의 z1=X@W1+b1; h1=relu(z1); out=h1@W2+b2 와 같은 흐름.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x); // 2 -> 8
        let x = self.relu.forward(x);    // 음수는 0으로
        self.linear2.forward(x)          // 8 -> 1
    }
}

fn main() {
    let device = Default::default();
    println!("=== Day 59-60: burn MLP for XOR ===\n");
 
    // ================================================================
    // 2) 데이터 준비 — XOR
    //   입력 4개, 각 (2,) / 정답 4개, 각 (1,)
    //   XOR: 두 입력이 다르면 1, 같으면 0
    // ================================================================
    let x: Tensor<MyBackend, 2> = Tensor::from_data(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        &device,
    );
    let y: Tensor<MyBackend, 2> = Tensor::from_data(
        [[0.0], [1.0], [1.0], [0.0]],
        &device,
    );

    // ================================================================
    // 3) 모델 + 옵티마이저 생성
    //   AdamConfig::init()로 옵티마이저를 만든다.
    //   Adam = 경사하강법(이전에 손으로 w -= lr*dw 하던 것)의 개선판.
    //          파라미터마다 학습 속도를 자동 조절해줘서 더 빨리 수렴.
    // ================================================================
    let mut model: Mlp<MyBackend> = Mlp::new(&device);
    let mut optimizer = AdamConfig::new().init();
    let lr = 0.05;

    // ================================================================
    // 4) 학습 루프 — forward -> loss -> backward -> step
    //   이 4단계가 핵심. 06에서 손으로 하던 걸 burn이 대신 해준다.
    // ================================================================
    for epoch in 0..=2000 {
        // (a) forward: 예측
        let pred = model.forward(x.clone());
 
        // (b) loss: MSE = 평균((pred - y)^2)
        let loss = MseLoss::new().forward(
            pred.clone(),
            y.clone(),
            burn::nn::loss::Reduction::Mean,
        );
 
        // (c) backward: 기울기 자동 계산
        let grads = loss.backward();
        // 기울기를 이 모델의 파라미터들과 연결
        let grads = GradientsParams::from_grads(grads, &model);
 
        // (d) step: 옵티마이저가 파라미터 업데이트 (w = w - lr*dw 의 개선판)
        model = optimizer.step(lr, model, grads);
 
        if epoch % 500 == 0 {
            let loss_val = loss.into_scalar();
            println!("epoch {epoch:>4}: loss = {loss_val:.4}");
        }
    }

    // ================================================================
    // 5) 최종 예측 확인
    //   학습이 잘 됐으면 [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0 근처.
    // ================================================================
    println!("\n최종 예측:");
    let pred = model.forward(x.clone());
    let pred_data = pred.into_data();
    let pred_vec = pred_data.to_vec::<f32>().unwrap();
    let inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    let answers = [0, 1, 1, 0];
    for i in 0..4 {
        println!(
            "  {:?} -> {:.3}  (정답 {})",
            inputs[i], pred_vec[i], answers[i]
        );
    }
 
    println!("\n-> 은닉층 + relu 덕분에 XOR(비선형)을 풀 수 있다.");
    println!("   forward→loss→backward→step 4단계가 학습의 전부.");
}