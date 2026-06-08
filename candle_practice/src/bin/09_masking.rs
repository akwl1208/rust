// ================================================================
// candle 예제 09: 어텐션 마스킹 with 브로드캐스트
//
// 어떨 때 쓰나: Transformer에서 어떤 위치는 보고 어떤 위치는 못 보게 막을 때.
// 대표적으로 causal mask: 토큰 i는 자기보다 미래(j > i)를 못 보게 한다.
// 핵심: (batch, seq, seq) 점수에 (seq, seq) 마스크 하나를 브로드캐스트.
// ================================================================
 
use candle_core::{Tensor, Device, DType, IndexOp, D};
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 09: 어텐션 마스킹 (broadcast) ===\n");
 
    let seq = 4usize;   // 시퀀스 길이 4
    let batch = 2usize; // 배치 2개
 
    // --- 가짜 어텐션 점수 (batch, seq, seq) ---
    // 실제론 Q@K^T 로 나오지만, 여기선 설명용으로 1로 채운다.
    let scores = Tensor::ones((batch, seq, seq), DType::F32, &dev)?;
    println!("scores shape: {:?}", scores.shape().dims());
 
    // --- causal mask 만들기 (seq, seq) ---
    // 위치 (i, j)에서 j > i 면 '미래'라서 가려야 함.
    //   가리는 방법: 그 자리에 아주 큰 음수(-inf 근사)를 더한다.
    //   그러면 softmax 후 확률이 0에 수렴 -> 사실상 무시됨.
    // 먼저 0/1 마스크를 손으로 만든다. 1 = 가림(미래), 0 = 허용.
    let mut mask_data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                mask_data[i * seq + j] = 1.0; // 미래 위치 표시
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq, seq), &dev)?; // (seq, seq)
    println!("\ncausal mask (1=가림):\n{mask}");

    // 1인 자리에 -1e9 를 넣을 수 있게 변환. (mask * -1e9)
    let neg = (mask * -1e9)?; // (seq, seq), 가릴 자리 = -1e9, 나머지 = 0
    println!("\n더해질 마스크 (가릴 곳 = 큰 음수):\n{neg}");

    // --- 브로드캐스트로 모든 배치에 같은 마스크 적용 ---
    // scores: (batch, seq, seq), neg: (seq, seq)
    //   뒤에서부터 맞추면 (seq,seq)는 그대로, batch 차원은 복사됨.
    let masked = scores.broadcast_add(&neg)?; // (batch, seq, seq)
    println!("\n마스킹된 scores (배치 0번):");
    println!("{}", masked.i(0)?);

    // --- softmax를 마지막 축에 직접 구현 ---
    // candle-nn 없이: exp(x - max) / sum(exp(x - max))
    //   max를 빼는 건 수치 안정성(overflow 방지) 때문. 표준 트릭.
    let max = masked.max_keepdim(D::Minus1)?;        // (batch, seq, 1)
    let shifted = masked.broadcast_sub(&max)?;        // 브로드캐스트로 max 빼기
    let exp = shifted.exp()?;                         // (batch, seq, seq)
    let sum = exp.sum_keepdim(D::Minus1)?;            // (batch, seq, 1)
    let probs = exp.broadcast_div(&sum)?;             // 브로드캐스트로 정규화
 
    println!("\nsoftmax 후 (배치 0번, 위 삼각형이 0에 가까움):");
    println!("{}", probs.i(0)?);

    Ok(())
}