// ================================================================
// Day 63-65: Attention을 candle로 구현 (numpy scratch 버전과 수치 비교)
//
// 목표: numpy로 이해한 softmax(QK^T/sqrt(d_k))V 를 candle 텐서로 옮기고,
//       같은 입력에서 numpy와 같은 숫자가 나오는지 확인한다.
//
// numpy 기준값(아래 주석)과 비교하며 보면 좋다.
//
// 실행:  cargo run --bin 11_attention
// ================================================================
 
use candle_core::{Tensor, Device, D};

// ----------------------------------------------------------------
// softmax를 마지막 축에 직접 구현 (candle-nn 의존성 없이)
//   exp(x - max) / sum(exp(x - max))  — max 빼는 건 overflow 방지
// ----------------------------------------------------------------
fn softmax_last_dim(x: &Tensor) -> candle_core::Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}

// ----------------------------------------------------------------
// Scaled Dot-Product Attention
//   수식: softmax(Q K^T / sqrt(d_k)) V
//   numpy의 scaled_dot_product_attention 과 1:1 대응
// ----------------------------------------------------------------
fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<(Tensor, Tensor)> {
    let d_k = q.dim(D::Minus1)? as f64;
 
    // 1) Q K^T : 마지막 두 축을 바꿔 전치 후 행렬곱
    //    numpy의 Q @ K.swapaxes(-1,-2) 와 같음
    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let scores = q.matmul(&k_t)?;
 
    // 2) / sqrt(d_k) : 스케일링
    let scores = (scores / d_k.sqrt())?;
 
    // 3) (선택) 마스크: 가릴 자리에 아주 큰 음수를 더함
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };
 
    // 4) softmax : 행마다 합=1인 확률
    let weights = softmax_last_dim(&scores)?;
 
    // 5) weights @ V : 가중합
    let output = weights.matmul(v)?;
    Ok((output, weights))
}

// causal mask (seq, seq): 미래 위치(j>i)에 -1e9, 나머지 0
fn make_causal_mask(seq: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                data[i * seq + j] = -1e9; // softmax 후 0이 되도록
            }
        }
    }
    Tensor::from_vec(data, (seq, seq), dev)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== Day 63-65: candle Attention ===\n");

    // numpy와 비교하기 위해 '고정' 입력 사용 (랜덤 아님)
    let q = Tensor::new(
        &[[1f32, 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]],
        &dev,
    )?;
    let k = Tensor::new(
        &[[1f32, 0., 1., 0.], [1., 1., 0., 0.], [0., 0., 1., 1.]],
        &dev,
    )?;
    let v = Tensor::new(
        &[[1f32, 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
        &dev,
    )?;

    // --- mask 없는 버전 ---
    let (out, weights) = scaled_dot_product_attention(&q, &k, &v, None)?;
    println!("attention weights (numpy와 비교):");
    println!("{weights}");
    // numpy 기준값:
    //  [[0.4519 0.2741 0.2741]
    //   [0.2327 0.3837 0.3837]
    //   [0.3072 0.5065 0.1863]]
    println!("\noutput:");
    println!("{out}");
    // numpy 기준값:
    //  [[4.2888 5.2888 6.2888 7.2888]
    //   [5.6038 6.6038 7.6038 8.6038]
    //   [4.5165 5.5165 6.5165 7.5165]]

    // --- causal mask 버전 ---
    let mask = make_causal_mask(3, &dev)?;
    let (out_c, weights_c) = scaled_dot_product_attention(&q, &k, &v, Some(&mask))?;
    println!("\ncausal weights (위 삼각형=0):");
    println!("{weights_c}");
    // numpy 기준값:
    //  [[1.     0.     0.    ]
    //   [0.3775 0.6225 0.    ]
    //   [0.3072 0.5065 0.1863]]
    println!("\ncausal output:");
    println!("{out_c}");

    println!("\n-> numpy 기준값(주석)과 숫자가 일치하면 구현 성공!");
    Ok(())
}