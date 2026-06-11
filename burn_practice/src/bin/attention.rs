use burn::{
    prelude::*,
    tensor::{activation::softmax, Bool, Tensor},
};

fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    mask: Option<Tensor<B, 2>>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // q shape: [seq, dim]
    // 예: [3, 4]
    let d_k = q.dims()[1] as f64;

    // 1. K를 뒤집기
    // Candle: k.transpose(D::Minus2, D::Minus1)?
    let k_t = k.swap_dims(0, 1);

    // 2. QK^T 계산
    // Candle: q.matmul(&k_t)?
    let scores = q.matmul(k_t);

    // 3. sqrt(d_k)로 나누기
    let scores = scores / d_k.sqrt();

    // 4. mask 있으면 더하기
    // Candle: scores.broadcast_add(m)?
    let scores = match mask {
        Some(m) => scores + m,
        None => scores,
    };

    // 5. softmax
    // Candle에서는 softmax_last_dim 직접 구현했음
    let weights = softmax(scores, 1);

    // 6. attention weight로 V 섞기
    let output = weights.clone().matmul(v);

    (output, weights)
}

fn make_causal_mask<B: Backend>(
    seq: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    // true인 위치에 -1e9를 채울 것
    let mask = Tensor::<B, 2, Bool>::tril_mask([seq, seq], 0, device);

    Tensor::<B, 2>::zeros([seq, seq], device)
        .mask_fill(mask, -1e9)
}

fn main() {
    type B = burn::backend::Wgpu;

    let device = Default::default();

    let q = Tensor::<B, 2>::from_floats(
        [
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [1., 1., 0., 0.],
        ],
        &device,
    );

    let k = Tensor::<B, 2>::from_floats(
        [
            [1., 0., 1., 0.],
            [1., 1., 0., 0.],
            [0., 0., 1., 1.],
        ],
        &device,
    );

    let v = Tensor::<B, 2>::from_floats(
        [
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
        ],
        &device,
    );

    let (out, weights) =
        scaled_dot_product_attention(q.clone(), k.clone(), v.clone(), None);

    println!("weights without mask:\n{weights}");
    println!("output without mask:\n{out}");

    let mask = make_causal_mask::<B>(3, &device);

    let (out_c, weights_c) =
        scaled_dot_product_attention(q, k, v, Some(mask));

    println!("weights with causal mask:\n{weights_c}");
    println!("output with causal mask:\n{out_c}");
}