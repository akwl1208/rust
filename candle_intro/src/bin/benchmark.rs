use candle_core::{Tensor, DType, Device};
use std::time::Instant;

fn bench(device: &Device, name: &str, size: usize) -> Result<(), Box<dyn std::error::Error>> {
    // size x size 행렬 두 개를 난수로 생성
    let a = Tensor::randn(0f32, 1f32, (size, size), device)?;
    let b = Tensor::randn(0f32, 1f32, (size, size), device)?;

    // 워밍업 1회 (첫 실행은 초기화 비용이 섞여 부정확)
    let _ = a.matmul(&b)?;
 
    // 실제 측정: matmul 10회
    let start = Instant::now();
    for _ in 0..10 {
        let c = a.matmul(&b)?;
        // GPU는 비동기라, 결과를 강제로 동기화해야 정확.
        // to_vec... 등으로 값을 꺼내면 실제 계산이 끝난다.
        let _ = c.sum_all()?.to_scalar::<f32>()?;
    }
    let elapsed = start.elapsed();
 
    println!("  [{name}] {size}x{size} matmul x10 = {:?}", elapsed);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CPU vs GPU matmul 속도 비교 ===\n");
 
    let size = 1000; // 1000x1000 행렬
 
    // CPU
    let cpu = Device::Cpu;
    bench(&cpu, "CPU", size)?;
 
    // GPU (있을 때만). new_cuda가 실패하면 GPU 없음.
    match Device::new_cuda(0) {
        Ok(gpu) => {
            bench(&gpu, "GPU", size)?;
            println!("\n-> 같은 코드, device만 바꿔 GPU에서 훨씬 빠르게.");
        }
        Err(_) => {
            println!("\n  (GPU(CUDA) 없음 — CPU만 측정. GPU가 있으면 수십 배 빠름)");
        }
    }
 
    println!("\n핵심: 측정은 반드시 'cargo run --release' 로!");
    println!("      debug 빌드는 최적화가 꺼져 있어 수십 배 느리다.");
    Ok(())
}