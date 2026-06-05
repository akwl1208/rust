// ================================================================
// candle 예제 03: 활성화 함수 (신경망의 비선형성)
//
// 어떨 때 쓰나: 신경망 각 층 뒤에서 비선형성을 준다.
// candle은 주요 활성화 함수를 메서드로 내장하고 있어 직접 안 짜도 된다.
// ================================================================
 
use candle_core::{Tensor, Device};
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    println!("=== 03: 활성화 함수 ===\n");
 
    // -3부터 3까지 입력
    let x = Tensor::new(&[-3.0f32, -1.0, 0.0, 1.0, 3.0], &dev)?;
    println!("입력 x: {:?}\n", x.to_vec1::<f32>()?);

    // relu: max(0,x). 음수는 0, 양수 그대로. 딥러닝 기본.
    println!("relu:  {:?}", x.relu()?.to_vec1::<f32>()?);

    // gelu: ReLU의 부드러운 버전. Transformer(GPT/BERT)가 주로 사용.
    //   gelu()는 tanh 근사, gelu_erf()는 정확한 erf 버전.
    println!("gelu:  {:?}", x.gelu()?.to_vec1::<f32>()?);

    // silu (= swish): x * sigmoid(x). LLaMA 등에서 사용.
    println!("silu:  {:?}", x.silu()?.to_vec1::<f32>()?);

    // tanh: -1~1 사이로. 옛날 RNN 등에서.
    println!("tanh:  {:?}", x.tanh()?.to_vec1::<f32>()?);

    // sigmoid는 메서드가 없을 수 있어 직접: 1/(1+exp(-x))
    //   neg() = 부호 반전, exp(), affine으로 +1, recip()=역수
    let sigmoid = ((x.neg()?.exp()? + 1.0)?).recip()?;
    println!("sigmoid: {:?}", sigmoid.to_vec1::<f32>()?);

    Ok(())
}