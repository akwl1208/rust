// =============================================================================
//  Day 66-67  미니 GPT 직접 조립하기 (Rust / candle 버전)
// =============================================================================
//
// [오늘의 학습 목표]
//   "블록을 직접 조립해봐야 '어디를 수정하면 어떤 효과가 나는가'를 알 수 있다."
//
// candle = Hugging Face가 만든 순수 Rust 딥러닝 라이브러리.
// PyTorch와 사용감이 비슷해서, 앞에서 본 Python(numpy) 코드와
// 거의 1:1로 대응됩니다. numpy 버전을 옆에 두고 비교하며 읽으세요.
//
// [Rust 초보자를 위한 메모]
//   - Tensor: 숫자 다차원 배열 (numpy의 ndarray 같은 것)
//   - Result<T>: 실패할 수 있는 연산. 끝에 '?'를 붙이면 에러를 위로 전달.
//   - VarBuilder: 학습 파라미터(가중치)를 만들고 보관하는 도구.
//   - &Tensor 처럼 '&'가 붙으면 '빌려쓰기'(소유권을 가져가지 않음)란 뜻.
// =============================================================================


use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder, VarMap};

// =============================================================================
// 1) RMSNorm  —  정규화 레이어
// =============================================================================
// 평균을 빼지 않고 RMS(제곱평균제곱근)로만 나눕니다. LayerNorm보다 빠름.
//   RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gain
struct RmsNorm {
    gain: Tensor, // 학습되는 게인 g, 모양 (d_model,)
    eps: f64,
}

impl RmsNorm {
    fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        // gain을 1로 초기화해서 생성 (정규화만 하고 시작)
        let gain = vb.get_with_hints(d_model, "gain", candle_nn::Init::Const(1.0))?;
        Ok(Self { gain, eps: 1e-6 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x 모양: (batch, seq, d_model)
        // 1) 제곱 → 마지막 차원(D::Minus1) 평균.  keepdim=true로 차원 유지
        let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?; // (batch, seq, 1)
        // 2) sqrt(평균 + eps) = RMS
        let rms = (mean_sq + self.eps)?.sqrt()?; // (batch, seq, 1)
        // 3) x / rms  (브로드캐스트로 마지막 차원에 나눠짐)
        let normed = x.broadcast_div(&rms)?;
        // 4) gain 곱하기 (gain은 (d_model,) → 브로드캐스트)
        normed.broadcast_mul(&self.gain)
    }
}

// =============================================================================
// 2) SwiGLU FFN  —  각 토큰을 변환하는 레이어
// =============================================================================
//   SwiGLU(x) = ( silu(x@W_gate) * (x@W_up) ) @ W_down
// 가중치 3개(gate, up, down)를 쓰고 게이팅(원소별 곱)을 합니다. bias는 없음.
struct SwiGluFfn {
    w_gate: Linear, // d_model -> d_ff
    w_up: Linear,   // d_model -> d_ff
    w_down: Linear, // d_ff   -> d_model
}

impl SwiGluFfn {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        // linear_no_bias = bias 없는 선형 레이어 (현대 LLM 관행)
        Ok(Self {
            w_gate: linear_no_bias(d_model, d_ff, vb.pp("w_gate"))?,
            w_up: linear_no_bias(d_model, d_ff, vb.pp("w_up"))?,
            w_down: linear_no_bias(d_ff, d_model, vb.pp("w_down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // silu(z) = z * sigmoid(z).  candle이 silu를 기본 제공.
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?; // (b, s, d_ff)
        let up = self.w_up.forward(x)?; // (b, s, d_ff)
        let fused = (gate * up)?; // 원소별 곱 (게이팅)
        self.w_down.forward(&fused) // (b, s, d_model)
    }
}

// =============================================================================
// 3) Multi-Head Attention  —  토큰끼리 정보 교환
// =============================================================================
// 각 토큰이 다른 토큰을 얼마나 주목할지 계산해서 정보를 가져옵니다.
// causal 마스크로 미래 토큰은 못 보게 막습니다 (GPT는 다음 토큰 예측).
struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    n_heads: usize,
    d_head: usize,
}

impl MultiHeadAttention {
    fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(d_model % n_heads == 0, "d_model은 n_heads로 나누어떨어져야 함");
        Ok(Self {
            // 여기서는 numpy 버전과 파라미터 수를 맞추려고 bias 없는 Linear 사용
            w_q: linear_no_bias(d_model, d_model, vb.pp("w_q"))?,
            w_k: linear_no_bias(d_model, d_model, vb.pp("w_k"))?,
            w_v: linear_no_bias(d_model, d_model, vb.pp("w_v"))?,
            w_o: linear_no_bias(d_model, d_model, vb.pp("w_o"))?,
            n_heads,
            d_head: d_model / n_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, d_model) = x.dims3()?;

        // 1) Q, K, V 계산
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // 2) head로 쪼개기: (b, seq, d_model) → (b, n_heads, seq, d_head)
        //    reshape 후 transpose(1,2)로 seq와 head 축을 맞바꿈
        let split = |t: &Tensor| -> Result<Tensor> {
            t.reshape((batch, seq, self.n_heads, self.d_head))?
                .transpose(1, 2)? // (b, n_heads, seq, d_head)
                .contiguous() // 메모리 정렬 (이후 matmul 위해 필요)
        };
        let q = split(&q)?;
        let k = split(&k)?;
        let v = split(&v)?;

        // 3) 주목 점수 = Q @ K^T / sqrt(d_head)
        let scale = (self.d_head as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?; // 마지막 두 축 전치
        let scores = (q.matmul(&k_t)? / scale)?; // (b, n_heads, seq, seq)

        // 4) causal 마스크: 미래 위치에 아주 작은 값(-1e9)을 '더해서'
        //    softmax를 통과하면 0이 되게 만듭니다.
        //    (where_cond 대신 덧셈 마스크를 쓰는 게 dtype 문제도 없고 표준적)
        let mask = causal_bias(seq, x.device())?; // (seq, seq): 미래=-1e9, 나머지=0
        let mask_b = mask.reshape((1, 1, seq, seq))?; // 브로드캐스트용 차원 추가
        let scores = scores.broadcast_add(&mask_b)?;

        // 5) softmax → V 가중합
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // (b, n_heads, seq, d_head)

        // 6) head 합치기: (b, n_heads, seq, d_head) → (b, seq, d_model)
        let out = out
            .transpose(1, 2)? // (b, seq, n_heads, d_head)
            .contiguous()?
            .reshape((batch, seq, d_model))?;

        // 7) 출력 변환
        self.w_o.forward(&out)
    }
}

// causal 바이어스 만들기: 미래 위치(j>i)는 -1e9, 나머지는 0.0
// 이 값을 점수에 '더하면' 미래 위치가 softmax 후 0이 됩니다.
fn causal_bias(seq: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                data[i * seq + j] = -1e9; // 미래 토큰 차단
            }
        }
    }
    Tensor::from_vec(data, (seq, seq), device)
}

// =============================================================================
// 4) Transformer 블록  —  오늘의 핵심! 조각들을 조립
// =============================================================================
// pre-norm 구조 + 잔차 연결(residual):
//   x = x + Attention(RMSNorm(x))
//   x = x + SwiGLU   (RMSNorm(x))
//
// ★ LoRA를 붙인다면 보통 attn의 w_q, w_v에 붙입니다.
//   이 구조가 보여야 "어디에 붙일지"가 보입니다.
struct TransformerBlock {
    norm1: RmsNorm,
    attn: MultiHeadAttention,
    norm2: RmsNorm,
    ffn: SwiGluFfn,
}

impl TransformerBlock {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::new(d_model, vb.pp("norm1"))?,
            attn: MultiHeadAttention::new(d_model, n_heads, vb.pp("attn"))?,
            norm2: RmsNorm::new(d_model, vb.pp("norm2"))?,
            ffn: SwiGluFfn::new(d_model, d_ff, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1단계: pre-norm → attention → 잔차 더하기
        let h = self.attn.forward(&self.norm1.forward(x)?)?;
        let x = (x + h)?;
        // 2단계: pre-norm → FFN → 잔차 더하기
        let h = self.ffn.forward(&self.norm2.forward(&x)?)?;
        x + h
    }
}

// =============================================================================
// 5) 미니 GPT  —  블록을 N개 쌓아서 완성
// =============================================================================
struct MiniGpt {
    token_emb: Embedding,         // 토큰 ID → 벡터
    blocks: Vec<TransformerBlock>, // 블록 N개
    final_norm: RmsNorm,
    lm_head: Linear,              // d_model → vocab
}

impl MiniGpt {
    fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let token_emb = embedding(vocab_size, d_model, vb.pp("token_emb"))?;

        // 블록 n_layers개 만들기
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // vb.pp(i) 로 각 블록마다 다른 이름공간을 줌 (가중치 충돌 방지)
            blocks.push(TransformerBlock::new(
                d_model,
                n_heads,
                d_ff,
                vb.pp("blocks").pp(i),
            )?);
        }

        let final_norm = RmsNorm::new(d_model, vb.pp("final_norm"))?;
        let lm_head = linear_no_bias(d_model, vocab_size, vb.pp("lm_head"))?;

        Ok(Self { token_emb, blocks, final_norm, lm_head })
    }

    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        // tokens 모양: (batch, seq), 정수(u32) 타입
        // 1) 임베딩 조회
        let mut x = self.token_emb.forward(tokens)?; // (b, s, d_model)
        // 2) 블록 통과
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        // 3) 최종 정규화
        x = self.final_norm.forward(&x)?;
        // 4) logits
        self.lm_head.forward(&x) // (b, s, vocab)
    }
}

// =============================================================================
// 실행
// =============================================================================
fn main() -> Result<()> {
    println!("============================================================");
    println!(" Day 66-67  미니 GPT 조립 (Rust/candle)");
    println!("============================================================");

    let device = Device::Cpu; // GPU 없이 CPU에서 실행

    // VarMap: 모든 가중치를 담는 그릇. VarBuilder는 거기에 이름 붙여 넣는 도구.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // [d_model=128, n_heads=4, n_layers=2] 커리큘럼 설정
    let vocab_size = 1000;
    let d_model = 128;
    let n_heads = 4;
    let n_layers = 2;
    let d_ff = 256;

    let model = MiniGpt::new(vocab_size, d_model, n_heads, n_layers, d_ff, vb)?;

    // 더미 입력: batch=2, seq=16. 토큰 ID는 u32 정수.
    let tokens = Tensor::from_vec(
        (0..2 * 16).map(|i| (i % vocab_size) as u32).collect::<Vec<_>>(),
        (2, 16),
        &device,
    )?;

    let logits = model.forward(&tokens)?;
    println!("\n입력 토큰 모양 : {:?}      (batch=2, seq=16)", tokens.dims());
    println!("출력 logits 모양: {:?}  (batch=2, seq=16, vocab=1000)", logits.dims());

    // 파라미터 수 세기: varmap에 들어있는 모든 텐서의 원소 수를 합산
    let mut total = 0usize;
    for (_name, var) in varmap.data().lock().unwrap().iter() {
        total += var.as_tensor().elem_count();
    }
    println!("\n총 파라미터 수 : {} 개", total);

    let emb = vocab_size * d_model;
    let head = d_model * vocab_size;
    println!("  - 임베딩    : {} 개", emb);
    println!("  - lm_head   : {} 개", head);
    println!("  - 위 둘 합  : {} 개  (전체의 큰 비중!)", emb + head);
    println!("\n  → 작은 모델에선 vocab 관련 파라미터가 절반 가까이를 차지합니다.");

    Ok(())
}