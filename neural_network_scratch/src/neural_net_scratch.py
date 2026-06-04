# ================================================================
# Day 53-54: Week 7 최종 프로젝트 — 신경망 scratch 구현 (Python)
#
# numpy만으로 완전한 신경망을 처음부터 구현합니다 (역전파 포함).
# 그동안 배운 모든 것이 여기서 합쳐집니다:
#   순전파(45-47) + 역전파(48-50) + 미니배치/정규화(51-52)
#
# 두 문제를 모두 학습:
#   XOR   : [2 -> 4 -> 1], sigmoid   (가장 작은 비선형 문제)
#   숫자분류: [64 -> 128 -> 10], ReLU+softmax (MNIST 스타일)
#
# 실행:  python3 neural_net_scratch.py
# ================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# 공통 활성화 함수
# ================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # 수치 안정화
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ================================================================
# 문제 1: XOR  [2 -> 4 -> 1]
# ================================================================
# XOR은 선형으로 못 푸는 가장 단순한 문제. 은닉층이 있어야 풀린다.
# (직선 하나로 (0,0),(1,1) 과 (0,1),(1,0) 을 가를 수 없음)
#   입력 2 -> 은닉 4 (sigmoid) -> 출력 1 (sigmoid)
#   손실: MSE

def train_xor():
    print("=" * 54)
    print(" 문제 1: XOR  [2 -> 4 -> 1]")
    print("=" * 54)
    print("XOR = 선형분리 불가능. 은닉층이 있어야 풀 수 있는 고전 문제.\n")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    rng = np.random.default_rng(42)
    W1 = rng.standard_normal((2, 4)) * 0.5
    b1 = np.zeros((1, 4))
    W2 = rng.standard_normal((4, 1)) * 0.5
    b2 = np.zeros((1, 1))
    lr = 0.5
    losses = []

    for ep in range(1, 5001):
        # --- 순전파 ---
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        loss = np.mean((a2 - y) ** 2)
        losses.append(loss)

        # --- 역전파 ---
        # MSE + sigmoid 출력: dL/dz2 = 2(a2-y)/N * a2(1-a2)
        dz2 = 2 * (a2 - y) / len(X) * a2 * (1 - a2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * a1 * (1 - a1)          # sigmoid 미분
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # --- 업데이트 ---
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

        if ep in (1, 1000, 3000, 5000):
            pred = (a2 > 0.5).astype(int)
            acc = (pred == y).mean()
            print(f"ep {ep:>4}: loss={loss:.5f}  acc={acc:.2f}  "
                  f"preds={a2.ravel().round(3)}")

    print(f"\n정답: {y.ravel()}")
    print("-> 초반엔 0.5 근처에서 헤매다가 어느 순간 풀린다 (XOR의 전형적 패턴)\n")
    return losses


# ================================================================
# 문제 2: 손글씨 숫자 분류  [64 -> 128 -> 10]  (MNIST 스타일)
# ================================================================
# sklearn의 digits: 8x8(=64픽셀) 손글씨 숫자 1797장, 10개 클래스.
# 진짜 MNIST(28x28)의 축소판 — 구조/원리가 완전히 동일하고
# 다운로드 없이 오프라인으로 즉시 학습 가능.
#   입력 64 -> 은닉 128 (ReLU) -> 출력 10 (softmax)
#   손실: cross-entropy

def train_digits():
    print("=" * 54)
    print(" 문제 2: 손글씨 숫자 분류  [64 -> 128 -> 10]")
    print("=" * 54)
    print("8x8 손글씨 숫자(MNIST 축소판), ReLU + softmax, cross-entropy\n")

    from sklearn.datasets import load_digits
    data = load_digits()
    X = data.data / 16.0                  # 0~16 -> 0~1 정규화
    y = data.target
    Y = np.eye(10)[y]                     # one-hot

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, te = idx[:1400], idx[1400:]       # 학습/테스트 분리
    Xtr, Ytr = X[tr], Y[tr]
    Xte, yte = X[te], y[te]

    # He 초기화 (ReLU에 적합)
    W1 = rng.standard_normal((64, 128)) * np.sqrt(2 / 64)
    b1 = np.zeros((1, 128))
    W2 = rng.standard_normal((128, 10)) * np.sqrt(2 / 128)
    b2 = np.zeros((1, 10))
    lr = 0.1
    losses, accs = [], []

    for ep in range(1, 101):
        # --- 순전파 ---
        z1 = Xtr @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        p = softmax(z2)
        loss = -np.mean(np.sum(Ytr * np.log(p + 1e-9), axis=1))
        losses.append(loss)

        # --- 역전파 ---
        # softmax + cross-entropy: dL/dz2 = (p - Y)/N  (깔끔!)
        dz2 = (p - Ytr) / len(Xtr)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)              # ReLU 미분
        dW1 = Xtr.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # --- 업데이트 ---
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

        # 테스트 정확도
        test_logits = relu(Xte @ W1 + b1) @ W2 + b2
        acc = (test_logits.argmax(axis=1) == yte).mean()
        accs.append(acc)

        if ep in (1, 10, 30, 50, 100):
            print(f"ep {ep:>3}: loss={loss:.4f}  test_acc={acc:.4f}")

    print(f"\n최종 테스트 정확도: {accs[-1]:.4f}  (무작위 추측이면 0.10)")
    print("-> 직접 만든 신경망이 손글씨 숫자를 90% 넘게 맞춘다!\n")
    return losses, accs


# ================================================================
# 시각화
# ================================================================

def plot_curves(xor_losses, digit_losses, digit_accs, out_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    ax1.plot(xor_losses, color="#7c3aed", linewidth=2)
    ax1.set_title("XOR: Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(digit_losses, color="#2563eb", linewidth=2)
    ax2.set_title("Digits: Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cross-Entropy")
    ax2.grid(True, alpha=0.3)

    ax3.plot(digit_accs, color="#16a34a", linewidth=2)
    ax3.set_title("Digits: Test Accuracy", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"학습 곡선 저장됨: {out_path}")


def main():
    print("\n" + "=" * 54)
    print(" Day 53-54: 신경망 scratch 구현 (최종 프로젝트)")
    print("=" * 54 + "\n")

    xor_losses = train_xor()
    digit_losses, digit_accs = train_digits()

    plot_curves(xor_losses, digit_losses, digit_accs, "learning_curves.png")

    print("=" * 54)
    print(" 완성!")
    print("=" * 54)
    print("numpy만으로 신경망 전체를 손으로 구현했다:")
    print("  순전파 -> 손실 -> 역전파 -> 업데이트, 그리고 학습 성공.")
    print("이제 PyTorch나 Rust의 burn을 쓸 때, 그 안에서 무슨 일이")
    print("일어나는지 안다. model.backward() 가 더 이상 마법이 아니다.")


if __name__ == "__main__":
    main()