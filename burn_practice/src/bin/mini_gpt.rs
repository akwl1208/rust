// =============================================================================
//  Day 66-67  미니 GPT 직접 조립하기 (Rust / burn 버전)
// =============================================================================
//
// [오늘의 학습 목표]
//   "블록을 직접 조립해봐야 '어디를 수정하면 어떤 효과가 나는가'를 알 수 있다."
//
// burn = Rust 네이티브 딥러닝 프레임워크.
// candle과 비교했을 때 burn의 특징:
//   - #[derive(Module)] : 구조체를 '학습 가능한 모듈'로 만들어줌
//                         (안에 든 가중치를 자동으로 추적/관리)
//   - #[derive(Config)] : 하이퍼파라미터 설정 구조체를 만들고
//                         .init(device) 로 실제 모듈을 생성하는 패턴
//   - 가중치를 일일이 이름 붙여 관리할 필요가 없음 (candle의 VarBuilder와 대비)
//
// [Rust 초보자 메모]
//   - <B: Backend> : 어떤 백엔드(CPU/GPU)에서 돌릴지를 일반화한 제네릭.
//                    "이 코드는 CPU든 GPU든 똑같이 동작한다"는 뜻.
//   - Tensor<B, 3> : 3차원 텐서 (예: batch, seq, d_model)
//   - .clone() : burn 텐서는 값이 아니라 '핸들'이라 clone이 가볍습니다.
// =============================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{activation::silu, backend::Backend, Int, Tensor},
};

// =============================================================================
// 1) RMSNorm  —  정규화 레이어
// =============================================================================
//   RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gain
// 평균을 빼지 않아 LayerNorm보다 빠릅니다.

// (1-a) 실제 모듈: 학습되는 가중치(gain)를 들고 있음
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    gain: burn::module::Param<Tensor<B, 1>>, // 학습되는 게인 g, 모양 (d_model,)
    eps: f64,
}

// (1-b) 설정 구조체: 어떻게 만들지에 대한 정보
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    d_model: usize,
    #[config(default = 1e-6)]
    eps: f64,
}

impl RmsNormConfig {
    // 설정 → 실제 모듈 생성
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm {
            // gain을 1로 초기화 (정규화만 하고 시작)
            gain: burn::module::Param::from_tensor(Tensor::ones([self.d_model], device)),
            eps: self.eps,
        }
    }
}

impl<B: Backend> RmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x 모양: (batch, seq, d_model)
        // 1) 제곱 → 마지막 차원(2번 축) 평균. mean_dim은 차원을 유지함 (keepdim).
        let mean_sq = x.clone().powf_scalar(2.0).mean_dim(2); // (batch, seq, 1)
        // 2) sqrt(평균 + eps) = RMS
        let rms = mean_sq.add_scalar(self.eps).sqrt(); // (batch, seq, 1)
        // 3) x / rms  (브로드캐스트로 마지막 차원에 나눠짐)
        let normed = x / rms;
        // 4) gain 곱하기. gain(d_model,)을 (1,1,d_model)로 펼쳐 브로드캐스트.
        let g = self.gain.val().reshape([1, 1, self.gain.val().dims()[0] as i32]);
        normed * g
    }
}

// =============================================================================
// 2) SwiGLU FFN  —  각 토큰을 변환하는 레이어
// =============================================================================
//   SwiGLU(x) = ( silu(x@W_gate) * (x@W_up) ) @ W_down
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    w_gate: Linear<B>, // d_model -> d_ff
    w_up: Linear<B>,   // d_model -> d_ff
    w_down: Linear<B>, // d_ff   -> d_model
}

#[derive(Config, Debug)]
pub struct SwiGluFfnConfig {
    d_model: usize,
    d_ff: usize,
}

impl SwiGluFfnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluFfn<B> {
        SwiGluFfn {
            // .with_bias(false) = bias 끄기 (현대 LLM 관행, 파라미터 절약)
            w_gate: LinearConfig::new(self.d_model, self.d_ff).with_bias(false).init(device),
            w_up: LinearConfig::new(self.d_model, self.d_ff).with_bias(false).init(device),
            w_down: LinearConfig::new(self.d_ff, self.d_model).with_bias(false).init(device),
        }
    }
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.w_gate.forward(x.clone())); // (b, s, d_ff) 게이트 신호
        let up = self.w_up.forward(x); // (b, s, d_ff) 값 신호
        self.w_down.forward(gate * up) // 원소별 곱(게이팅) 후 다시 d_model로
    }
}

// =============================================================================
// 3) Transformer 블록  —  오늘의 핵심! 조립
// =============================================================================
// pre-norm + 잔차 연결:
//   x = x + Attention(RMSNorm(x))
//   x = x + SwiGLU   (RMSNorm(x))
//
// Attention은 burn이 제공하는 MultiHeadAttention을 씁니다.
// (causal 마스크는 MhaInput에 옵션으로 줄 수 있지만, 여기서는 구조 이해가
//  목적이라 기본 self-attention으로 둡니다. 실제 학습 시엔 마스크를 켜세요.)
//
// ★ LoRA를 붙인다면 보통 attn 내부의 query/value 가중치에 붙입니다.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: RmsNorm<B>,
    attn: MultiHeadAttention<B>,
    norm2: RmsNorm<B>,
    ffn: SwiGluFfn<B>,
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            norm1: RmsNormConfig::new(self.d_model).init(device),
            attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads).init(device),
            norm2: RmsNormConfig::new(self.d_model).init(device),
            ffn: SwiGluFfnConfig::new(self.d_model, self.d_ff).init(device),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1단계: pre-norm → attention → 잔차 더하기
        let normed = self.norm1.forward(x.clone());
        // MhaInput::self_attn = query=key=value 모두 같은 입력 (self-attention)
        let attn_out = self.attn.forward(MhaInput::self_attn(normed)).context;
        let x = x + attn_out;

        // 2단계: pre-norm → FFN → 잔차 더하기
        let normed = self.norm2.forward(x.clone());
        let ffn_out = self.ffn.forward(normed);
        x + ffn_out
    }
}

// =============================================================================
// 4) 미니 GPT  —  블록을 N개 쌓아서 완성
// =============================================================================
#[derive(Module, Debug)]
pub struct MiniGpt<B: Backend> {
    token_emb: Embedding<B>,           // 토큰 ID → 벡터
    blocks: Vec<TransformerBlock<B>>,  // 블록 N개 (Vec도 Module이 됨!)
    final_norm: RmsNorm<B>,
    lm_head: Linear<B>,                // d_model → vocab
}

#[derive(Config, Debug)]
pub struct MiniGptConfig {
    vocab_size: usize,
    #[config(default = 128)]
    d_model: usize,
    #[config(default = 4)]
    n_heads: usize,
    #[config(default = 2)]
    n_layers: usize,
    #[config(default = 256)]
    d_ff: usize,
}

impl MiniGptConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniGpt<B> {
        // 블록 n_layers개 만들기
        let blocks = (0..self.n_layers)
            .map(|_| TransformerBlockConfig::new(self.d_model, self.n_heads, self.d_ff).init(device))
            .collect();

        MiniGpt {
            token_emb: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            blocks,
            final_norm: RmsNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).with_bias(false).init(device),
        }
    }
}

impl<B: Backend> MiniGpt<B> {
    // tokens: (batch, seq) 정수 인덱스
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // 1) 임베딩 조회
        let mut x = self.token_emb.forward(tokens); // (b, s, d_model)
        // 2) 블록 통과
        for block in &self.blocks {
            x = block.forward(x);
        }
        // 3) 최종 정규화
        let x = self.final_norm.forward(x);
        // 4) logits
        self.lm_head.forward(x) // (b, s, vocab)
    }
}

// =============================================================================
// 실행
// =============================================================================
fn main() {
    // 백엔드 지정: ndarray = CPU에서 도는 순수 Rust 백엔드
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;

    println!("============================================================");
    println!(" Day 66-67  미니 GPT 조립 (Rust/burn)");
    println!("============================================================");

    let device = NdArrayDevice::default();

    // [d_model=128, n_heads=4, n_layers=2] 커리큘럼 설정
    let config = MiniGptConfig::new(1000) // vocab_size=1000
        .with_d_model(128)
        .with_n_heads(4)
        .with_n_layers(2)
        .with_d_ff(256);

    let model = config.init::<B>(&device);

    // 더미 입력: batch=2, seq=16. 0~999 토큰 ID.
    let data: Vec<i32> = (0..2 * 16).map(|i| (i % 1000) as i32).collect();
    let tokens = Tensor::<B, 1, Int>::from_data(data.as_slice(), &device).reshape([2, 16]);

    let logits = model.forward(tokens);
    println!("\n입력 토큰 모양 : [2, 16]      (batch=2, seq=16)");
    println!("출력 logits 모양: {:?}  (batch=2, seq=16, vocab=1000)", logits.dims());

    // 파라미터 수.
    // burn Module은 num_params()를 제공하지만, 일부 버전/백엔드에서
    // 값이 안 맞는 경우가 있어 직접 세는 방법도 함께 보여줍니다.
    // (학습목표인 'params_count'를 확실히 이해하기 위해)
    println!("\n총 파라미터 수 (num_params): {} 개", model.num_params());

    // 손으로 계산한 기댓값 (검산용):
    //   임베딩         : 1000*128                 = 128,000
    //   블록당:
    //     RMSNorm x2   : 128*2                    =     256
    //     Attention    : 4*(128*128)              =  65,536  (bias 없음 가정)
    //     SwiGLU       : 2*(128*256) + 256*128    =  98,304
    //   블록 합계      : 164,096  → x2층          = 328,192
    //   최종 norm      : 128
    //   lm_head        : 128*1000                 = 128,000
    //   ----------------------------------------------------
    //   합계 ≈ 584,320
    let expected = 1000 * 128                       // 임베딩
        + 2 * (128 * 2 + 4 * 128 * 128 + 2 * 128 * 256 + 256 * 128) // 블록 2개
        + 128                                       // 최종 norm
        + 128 * 1000;                               // lm_head
    println!("손계산 기댓값            : {} 개", expected);

    let emb = 1000 * 128;
    let head = 128 * 1000;
    println!("  - 임베딩    : {} 개", emb);
    println!("  - lm_head   : {} 개", head);
    println!("  - 위 둘 합  : {} 개  (전체의 큰 비중!)", emb + head);
    println!("\n  → 작은 모델에선 vocab 관련 파라미터가 절반 가까이를 차지합니다.");
}