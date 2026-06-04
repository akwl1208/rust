# ================================================================
# Day 51-52: 배치 처리 & 정규화 기법 (Python/numpy)
#
# LLM을 실제로 학습 가능하게 만드는 실전 기법들:
#   1. 미니배치    - 데이터를 [batch, features]로 묶어 처리
#   2. LayerNorm   - 각 샘플을 정규화 (LLM의 표준, PyTorch와 비교)
#   3. BatchNorm   - 비교용: 정규화 '방향'의 차이
#   4. Dropout     - 과적합 방지 (학습/추론 모드 구분)
#   5. Grad Clip   - 폭발하는 gradient 제어
#
# "왜 LLaMA에 LayerNorm이 있는가"를 코드로 이해하는 게 목표.
#
# 실행:  python3 normalization.py
# ================================================================

import numpy as np


# ----------------------------------------------------------------
# 1. 미니배치와 shape
# ----------------------------------------------------------------
# 데이터를 하나씩 처리하지 않고 여러 개를 묶어 [batch, features]
# 형태로 한 번에 처리한다.
#   - 메모리/연산 효율 (행렬 곱으로 한 방에)
#   - 학습 안정성 (여러 샘플 평균 방향으로 업데이트)
#
# shape 표기: [batch, features]
#   batch=2, features=4 면 2x4 행렬.

def demo_minibatch():
    print("-- 1) 미니배치와 shape --\n")
    batch = np.array([
        [1.0, 2.0, 3.0, 4.0],   # 샘플 1
        [2.0, 4.0, 6.0, 8.0],   # 샘플 2
    ])
    print(f"미니배치 shape = {batch.shape}  = [batch={batch.shape[0]}, features={batch.shape[1]}]")
    print(f"샘플 1: {batch[0]}")
    print(f"샘플 2: {batch[1]}")
    print("-> 여러 샘플을 한 행렬로 묶어 한 번에 처리 (효율 + 안정성)\n")
    return batch


# ----------------------------------------------------------------
# 2. LayerNorm (LLM의 표준 정규화)
# ----------------------------------------------------------------
# 각 '샘플(행)' 안에서 feature들을 평균0 분산1로 정규화한 뒤,
# 학습 가능한 gamma(스케일), beta(이동)로 다시 조정한다.
#
#   mean, var = 행별 평균/분산 (feature 축으로)
#   x_norm = (x - mean) / sqrt(var + eps)
#   out = gamma * x_norm + beta
#
# 왜 LLM은 LayerNorm인가?
#   - 배치 크기에 의존하지 않는다 (샘플 하나씩도 정규화 가능)
#   - 시퀀스 길이가 제각각인 언어 데이터에 적합
#   - 각 층의 입력 분포를 안정시켜 깊은 신경망 학습을 가능케 함
#     (이게 없으면 값이 점점 커지거나 작아져 학습이 망가진다)

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=1, keepdims=True)   # 행별 평균 (feature 축)
    var = x.var(axis=1, keepdims=True)     # 행별 분산
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def demo_layernorm(batch):
    print("-- 2) LayerNorm (각 샘플을 정규화) --\n")
    gamma = np.ones(4)   # 스케일 (학습되는 파라미터, 여기선 1)
    beta = np.zeros(4)   # 이동  (학습되는 파라미터, 여기선 0)
    out = layernorm(batch, gamma, beta)

    print("입력:")
    print(batch)
    print("\nLayerNorm 출력 (각 행이 평균0 분산1로):")
    print(out.round(4))
    print(f"\n각 행의 평균: {out.mean(axis=1).round(6)}  (≈0)")
    print(f"각 행의 분산: {out.var(axis=1).round(4)}  (≈1)")
    print("-> 샘플마다 독립적으로 정규화. 배치 크기와 무관 -> LLM에 적합\n")
    return out


# ----------------------------------------------------------------
# 3. BatchNorm (비교용) — 정규화 '방향'이 다르다
# ----------------------------------------------------------------
# BatchNorm은 각 'feature(열)'를 배치 전체에 걸쳐 정규화한다.
#   -> 배치 안 여러 샘플의 같은 feature끼리 묶어 정규화
#   문제: 배치가 작거나 1이면 통계가 불안정. 시퀀스 데이터에 부적합.
#   그래서 LLM은 BatchNorm 대신 LayerNorm을 쓴다.

def batchnorm(x, eps=1e-5):
    mean = x.mean(axis=0, keepdims=True)   # 열별 평균 (배치 축)
    var = x.var(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def demo_batchnorm(batch):
    print("-- 3) BatchNorm (각 feature를 정규화) — 비교용 --\n")
    out = batchnorm(batch)
    print("BatchNorm 출력:")
    print(out.round(4))
    print(f"\n각 열의 평균: {out.mean(axis=0).round(6)}  (≈0, 열 기준!)")
    print()
    print("핵심 차이:")
    print("  LayerNorm: 행(샘플) 방향 정규화 -> 배치 크기 무관 -> LLM 표준")
    print("  BatchNorm: 열(feature) 방향 정규화 -> 배치에 의존 -> CNN 등에서 사용")
    print()


# ----------------------------------------------------------------
# 4. Dropout (과적합 방지) — 학습 vs 추론 모드
# ----------------------------------------------------------------
# 학습 중에 뉴런 일부를 무작위로 0으로 꺼서, 특정 뉴런에
# 과하게 의존하는 것을 막는다 (과적합 방지).
#
#   학습 모드: 확률 p로 끄고, 살아남은 값은 1/(1-p)배로 키움
#              (꺼진 만큼 보정 -> 출력 기댓값 유지. inverted dropout)
#   추론 모드: 아무것도 끄지 않음 (전체 뉴런 사용)
#
# 이 '모드 구분'이 핵심. 학습 때만 끄고 실제 예측 땐 전부 쓴다.

def dropout(x, p, training, rng):
    if not training or p == 0.0:
        return x  # 추론 모드: 그대로 통과
    mask = (rng.random(x.shape) > p) / (1.0 - p)  # 살아남은 것 보정
    return x * mask


def demo_dropout():
    print("-- 4) Dropout (학습 vs 추론 모드) --\n")
    rng = np.random.default_rng(0)
    x = np.ones(10)
    print(f"입력: {x}")
    print(f"학습 모드 (p=0.5): {dropout(x, 0.5, True, rng).round(2)}")
    print("                   ^ 일부는 0(꺼짐), 나머지는 2배(보정)")
    print(f"추론 모드 (p=0.5): {dropout(x, 0.5, False, rng).round(2)}")
    print("                   ^ 전부 그대로 (끄지 않음)")
    print()
    # 기댓값 보존 확인
    samples = [dropout(np.ones(1000), 0.5, True, rng).mean() for _ in range(200)]
    print(f"학습 모드 출력 평균(200회 반복): {np.mean(samples):.4f}")
    print("-> 입력 평균 1.0과 거의 같다. 스케일 보정 덕에 기댓값이 유지됨.")
    print("   (그래서 학습/추론 사이 출력 크기가 일관됨)\n")


# ----------------------------------------------------------------
# 5. Gradient Clipping (gradient 폭발 제어)
# ----------------------------------------------------------------
# 학습 중 gradient가 갑자기 커지면(폭발) 가중치가 엉뚱하게 튀어
# 학습이 망가진다. gradient의 전체 크기(norm)가 임계값을 넘으면
# 비율을 유지한 채 임계값으로 줄인다 (방향은 보존, 크기만 제한).
#
#   norm = sqrt(sum(g^2))
#   norm > max_norm 이면:  g = g * (max_norm / norm)
#
# LLM 학습에서 거의 필수. 이게 없으면 가끔 loss가 NaN으로 터진다.

def clip_grad(g, max_norm):
    norm = np.sqrt(np.sum(g ** 2))
    if norm > max_norm:
        g = g * (max_norm / norm)
    return g, norm


def demo_gradient_clipping():
    print("-- 5) Gradient Clipping (폭발 제어) --\n")
    for g in [np.array([3.0, 4.0]), np.array([0.3, 0.4])]:
        clipped, norm = clip_grad(g.copy(), max_norm=1.0)
        new_norm = np.sqrt(np.sum(clipped ** 2))
        status = "제한됨" if norm > 1.0 else "그대로 (임계값 이하)"
        print(f"gradient {g}, norm={norm:.2f} -> clip 후 {clipped.round(3)}, norm={new_norm:.3f}  [{status}]")
    print()
    print("-> 방향은 유지하고 크기만 임계값으로 제한. LLM 학습 안정성의 핵심.\n")


def main():
    print("=" * 54)
    print(" Day 51-52: 배치 처리 & 정규화 기법")
    print("=" * 54 + "\n")

    batch = demo_minibatch()
    demo_layernorm(batch)
    demo_batchnorm(batch)
    demo_dropout()
    demo_gradient_clipping()

    # PyTorch 비교 검증 (있으면)
    try:
        import torch
        import torch.nn as nn
        x32 = batch.astype(np.float32)
        ln = nn.LayerNorm(4)
        with torch.no_grad():
            ln.weight.fill_(1.0)
            ln.bias.fill_(0.0)
        pt = ln(torch.tensor(x32)).detach().numpy()
        mine = layernorm(x32, np.ones(4, np.float32), np.zeros(4, np.float32))
        print("=" * 54)
        print(" PyTorch LayerNorm 비교 검증")
        print("=" * 54)
        print(f"최대 오차: {np.abs(mine - pt).max():.2e}")
        print(f"일치? -> {'성공! 직접 구현이 PyTorch와 동일' if np.allclose(mine, pt, atol=1e-5) else '불일치'}")
    except ImportError:
        print("(torch 미설치 — PyTorch 비교는 건너뜀)")


if __name__ == "__main__":
    main()