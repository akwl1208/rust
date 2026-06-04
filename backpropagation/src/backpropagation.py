# ================================================================
# Day 48-50: 역전파(Backpropagation) 완전 이해 (Python/numpy)
#
# Day 45-47의 [3->4->2] 순전파에 '역전파'를 붙입니다.
# 순전파 -> 손실 -> 역전파 -> 가중치 업데이트 전체 루프를 구현하고,
# 역전파가 맞는지 '수치 미분'으로 검증합니다(gradient check).
#
# 핵심 질문: "각 가중치가 손실에 얼마나 기여했는가?"
#   역전파 = 이 질문에 연쇄 법칙으로 답하는 과정.
#   dL/dW = (출력 쪽 기울기) x (그 가중치가 출력에 준 영향)
#
# 'Autograd가 이걸 자동으로 해준다'의 의미를 손으로 확인하는 게 목표.
#
# 실행:  python3 backprop.py
# ================================================================

import numpy as np


# ----------------------------------------------------------------
# 신경망 [3 -> 4 -> 2], 은닉층 ReLU, 출력층 선형, 손실 MSE
# ----------------------------------------------------------------
# 계산 그래프(computation graph):
#
#   x --(W1,b1)--> z1 --ReLU--> a1 --(W2,b2)--> z2 --MSE--> loss
#
# 역전파는 이 그래프를 '거꾸로' 따라가며 각 단계의 기울기를 구한다.

def init_params():
    # 재현 가능하도록 고정된 값 사용
    W1 = np.array([[0.1, 0.2, -0.3],
                   [0.4, -0.5, 0.6],
                   [-0.7, 0.8, 0.1],
                   [0.2, 0.3, -0.4]])          # 4x3
    b1 = np.array([0.1, -0.2, 0.3, 0.0])       # 4
    W2 = np.array([[0.5, -0.6, 0.7, 0.1],
                   [-0.2, 0.3, -0.4, 0.8]])     # 2x4
    b2 = np.array([0.05, -0.05])               # 2
    return W1, b1, W2, b2


# ----------------------------------------------------------------
# 순전파 (forward) — 중간값(cache)을 저장해 역전파에서 재사용
# ----------------------------------------------------------------

def forward(x, target, params):
    W1, b1, W2, b2 = params
    z1 = W1 @ x + b1            # 은닉층 가중합
    a1 = np.maximum(0, z1)      # ReLU
    z2 = W2 @ a1 + b2           # 출력층 (선형)
    loss = np.mean((z2 - target) ** 2)  # MSE
    cache = (x, z1, a1, z2)
    return loss, z2, cache


# ----------------------------------------------------------------
# 역전파 (backward) — 손으로 전개한 연쇄 법칙을 그대로 코드로
# ----------------------------------------------------------------
# 손실에서 시작해 입력 방향으로 '거꾸로' 기울기를 전파한다.
#
# 1) 손실 -> z2:   L = mean((z2-t)^2) = (1/N) Σ(z2-t)^2
#       dL/dz2 = 2(z2 - t) / N
#
# 2) z2 = W2@a1 + b2:
#       dL/dW2 = dL/dz2 (바깥곱) a1      ← 기울기 x 입력
#       dL/db2 = dL/dz2
#       dL/da1 = W2^T @ dL/dz2           ← 기울기를 이전 층으로 전달
#
# 3) a1 = ReLU(z1):
#       dL/dz1 = dL/da1 * ReLU'(z1)
#       (ReLU 미분: z1>0 이면 1, 아니면 0  → 음수였던 뉴런은 기울기 차단)
#
# 4) z1 = W1@x + b1:
#       dL/dW1 = dL/dz1 (바깥곱) x
#       dL/db1 = dL/dz1

def backward(target, cache, params):
    x, z1, a1, z2 = cache
    W1, b1, W2, b2 = params
    N = len(z2)  # MSE 평균에 쓰인 출력 개수

    # 출력층
    dz2 = 2 * (z2 - target) / N        # dL/dz2
    dW2 = np.outer(dz2, a1)            # dL/dW2
    db2 = dz2                          # dL/db2

    # 은닉층으로 전파
    da1 = W2.T @ dz2                   # dL/da1
    dz1 = da1 * (z1 > 0)              # dL/dz1  (ReLU 미분 적용)
    dW1 = np.outer(dz1, x)            # dL/dW1
    db1 = dz1                          # dL/db1

    return dW1, db1, dW2, db2


# ----------------------------------------------------------------
# Gradient Check — 역전파가 맞는지 수치 미분으로 검증
# ----------------------------------------------------------------
# 수치 미분(중심차분): dL/dw ≈ [L(w+h) - L(w-h)] / 2h
# 역전파로 구한 기울기와 이게 1e-5 이내로 같으면 역전파가 정확한 것.

def gradient_check(x, target, params):
    W1, b1, W2, b2 = params
    loss, _, cache = forward(x, target, params)
    grads = backward(target, cache, params)
    analytic = {"W1": grads[0], "b1": grads[1], "W2": grads[2], "b2": grads[3]}

    h = 1e-5
    max_diff = 0.0
    for name, param in zip(["W1", "b1", "W2", "b2"], params):
        flat = param.ravel()
        g_flat = analytic[name].ravel()
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + h
            lp, _, _ = forward(x, target, params)
            flat[i] = orig - h
            lm, _, _ = forward(x, target, params)
            flat[i] = orig
            num = (lp - lm) / (2 * h)        # 수치 미분
            diff = abs(num - g_flat[i])       # 역전파와의 차이
            max_diff = max(max_diff, diff)

    return max_diff


# ----------------------------------------------------------------
# 메인: 검증 -> 학습 루프
# ----------------------------------------------------------------

def main():
    print("=" * 52)
    print(" Day 48-50: 역전파 완전 이해")
    print("=" * 52)
    print("신경망 [3->4->2], 은닉층 ReLU, 손실 MSE")
    print("계산그래프: x -(W1)-> z1 -ReLU-> a1 -(W2)-> z2 -MSE-> loss\n")

    x = np.array([1.0, 2.0, -1.0])
    target = np.array([1.0, 0.0])

    # --- 1) Gradient Check (학습 전에 역전파가 맞는지 먼저 검증) ---
    print("-- 1) Gradient Check (역전파 vs 수치미분) --\n")
    params = list(init_params())
    max_diff = gradient_check(x, target, params)
    print(f"해석적 기울기(역전파) vs 수치 미분 최대 오차: {max_diff:.2e}")
    print(f"1e-5 이내인가? -> {'성공! 역전파가 정확함' if max_diff < 1e-5 else '실패'}")
    print("(수치미분은 '느리지만 확실한' 정답. 역전파가 이와 같으면 OK)\n")

    # --- 2) 학습 루프 ---
    print("-- 2) 학습 루프 (순전파->손실->역전파->업데이트) --\n")
    params = list(init_params())
    lr = 0.1
    print(f"{'step':>6} {'loss':>12} {'output':>22}")
    print("-" * 42)
    for step in range(1, 201):
        loss, out, cache = forward(x, target, params)        # 순전파+손실
        dW1, db1, dW2, db2 = backward(target, cache, params)  # 역전파
        # 가중치 업데이트 (경사하강법)
        params[0] -= lr * dW1
        params[1] -= lr * db1
        params[2] -= lr * dW2
        params[3] -= lr * db2
        if step in (1, 10, 50, 100, 200):
            print(f"{step:>6} {loss:>12.6f}   [{out[0]:>8.4f}, {out[1]:>8.4f}]")

    print(f"\n목표(target):           [{target[0]:>8.4f}, {target[1]:>8.4f}]")
    print("-> 역전파로 구한 기울기 방향으로 가중치를 옮기니 출력이 정답에 수렴!")
    print("   이 '순전파->역전파->업데이트' 루프가 모든 신경망 학습의 핵심.")
    print("   Autograd(candle/burn/PyTorch)는 backward()를 자동 생성해줄 뿐,")
    print("   원리는 방금 손으로 짠 이 연쇄 법칙 그대로다.")


if __name__ == "__main__":
    main()