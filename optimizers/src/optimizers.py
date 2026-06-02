# ================================================================
# Day 44: 옵티마이저 심화 — SGD vs Momentum vs Adam vs AdamW
#
# "왜 LLM 파인튜닝은 AdamW를 표준으로 쓰는가?"를 코드로 체감합니다.
#
# 발전 흐름:
#   SGD       -> 모든 파라미터에 같은 학습률 (단순하지만 느리고 진동)
#   Momentum  -> 이전 방향을 기억해 진동을 줄이고 가속
#   Adam      -> 파라미터마다 학습률을 자동 조정 (Momentum + 적응형)
#   AdamW     -> Adam + Weight Decay (과적합 방지) = LLM 파인튜닝 표준
#
# 실행:  python3 optimizers.py
# ================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
# 테스트용 손실함수: f(x, y) = x^2 + 10*y^2
# ----------------------------------------------------------------
# 일부러 '비대칭 계곡' 모양으로 만들었다.
#   y 방향은 경사가 매우 가파르고(계수 10), x 방향은 완만하다(계수 1).
# 이런 지형에서 옵티마이저 간 차이가 극명하게 드러난다:
#   - SGD: 가파른 y 방향에서 좌우로 진동하며 느리게 내려감
#   - Momentum/Adam: 진동을 억누르고 빠르게 골짜기로
#
# 최저점은 (0, 0), 그때 손실 = 0.

def loss(p):
    x, y = p
    return x ** 2 + 10 * y ** 2

def grad(p):
    x, y = p
    return np.array([2 * x, 20 * y])  # [df/dx, df/dy]


START = np.array([5.0, 5.0])  # 모두 같은 출발점
STEPS = 60


# ----------------------------------------------------------------
# 1. SGD (Stochastic Gradient Descent)
# ----------------------------------------------------------------
# 가장 단순. 기울기 반대 방향으로 일정 보폭(lr)만큼 이동.
#   p <- p - lr * grad
#
# 문제: 모든 파라미터에 '같은' 학습률을 쓴다.
#   가파른 방향엔 너무 크고(진동), 완만한 방향엔 너무 작다(느림).

def sgd(lr=0.01):
    p = START.copy()
    history = [loss(p)]
    for _ in range(STEPS):
        p = p - lr * grad(p)
        history.append(loss(p))
    return history


# ----------------------------------------------------------------
# 2. Momentum
# ----------------------------------------------------------------
# '관성'을 도입. 이전 이동 방향(속도 v)을 기억해 누적한다.
#   v <- beta * v + grad      (이전 속도를 beta만큼 유지 + 새 기울기)
#   p <- p - lr * v
#
# 효과: 같은 방향으로 계속 가면 가속(공이 비탈 굴러내리듯),
#       좌우로 왔다갔다 하는 진동은 상쇄되어 줄어든다.

def momentum(lr=0.01, beta=0.9):
    p = START.copy()
    v = np.zeros(2)
    history = [loss(p)]
    for _ in range(STEPS):
        v = beta * v + grad(p)
        p = p - lr * v
        history.append(loss(p))
    return history


# ----------------------------------------------------------------
# 3. Adam (Adaptive Moment Estimation)
# ----------------------------------------------------------------
# Momentum + '파라미터마다 다른 학습률'을 합친 것.
#   m: 기울기의 평균 (1차 모멘트) = 방향 (Momentum 역할)
#   v: 기울기 제곱의 평균 (2차 모멘트) = 그 파라미터가 얼마나 출렁였나
#
#   m <- b1*m + (1-b1)*g
#   v <- b2*v + (1-b2)*g^2
#   (초반 편향 보정)  m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)
#   p <- p - lr * m_hat / (sqrt(v_hat) + eps)
#
# 핵심: 많이 출렁인 파라미터(v 큼)는 학습률을 자동으로 줄이고,
#       잠잠한 파라미터(v 작음)는 학습률을 키운다 -> 파라미터별 적응.
#   b1=0.9, b2=0.999 는 사실상 표준 기본값.

def adam(lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    p = START.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    history = [loss(p)]
    for t in range(1, STEPS + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)   # 초반 편향 보정
        v_hat = v / (1 - b2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(loss(p))
    return history


# ----------------------------------------------------------------
# 4. AdamW (Adam + Weight Decay) — LLM 파인튜닝 표준
# ----------------------------------------------------------------
# Adam에 'weight decay'를 더한 것. weight decay = 가중치를 매 스텝
# 조금씩 0쪽으로 끌어당겨, 가중치가 너무 커지는 것(과적합)을 막는다.
#
# AdamW의 핵심 (논문의 포인트):
#   기존 Adam은 weight decay를 gradient에 섞어 넣었는데,
#   그러면 적응형 학습률에 의해 decay 효과가 왜곡된다.
#   AdamW는 decay를 gradient와 '분리'해서 파라미터에 직접 적용한다:
#
#   p <- p - lr * (m_hat / (sqrt(v_hat)+eps))  - lr * wd * p
#                  └─ Adam 업데이트 ─┘          └─ 분리된 decay ─┘
#
# 이 'decoupled(분리된) weight decay'가 AdamW라는 이름의 W.

def adamw(lr=0.3, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
    p = START.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    history = [loss(p)]
    params_norm = [np.linalg.norm(p)]
    for t in range(1, STEPS + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        # Adam 업데이트 + 분리된 weight decay
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * wd * p
        history.append(loss(p))
        params_norm.append(np.linalg.norm(p))
    return history, params_norm


# ----------------------------------------------------------------
# 실험 A: SGD vs Momentum vs Adam 수렴 속도 비교
# ----------------------------------------------------------------

def experiment_convergence():
    print("=" * 56)
    print(" 실험 A: 수렴 속도 비교 (SGD vs Momentum vs Adam)")
    print("=" * 56)
    print("손실함수 f(x,y) = x^2 + 10y^2, 출발점 (5,5), 최저점 (0,0)\n")

    h_sgd = sgd(lr=0.01)
    h_mom = momentum(lr=0.01)
    h_adam = adam(lr=0.3)

    print(f"{'step':>6} {'SGD':>14} {'Momentum':>14} {'Adam':>14}")
    print("-" * 50)
    for s in [0, 5, 10, 20, 40, 60]:
        print(f"{s:>6} {h_sgd[s]:>14.6f} {h_mom[s]:>14.6f} {h_adam[s]:>14.6f}")
    print()
    print(f"최종 손실:  SGD={h_sgd[-1]:.6f}  Momentum={h_mom[-1]:.6f}  Adam={h_adam[-1]:.6f}")
    print("-> SGD는 진동하며 느리게, Momentum은 가속, Adam은 적응형으로 안정 수렴\n")

    return h_sgd, h_mom, h_adam


# ----------------------------------------------------------------
# 실험 B: AdamW의 weight decay 효과
# ----------------------------------------------------------------

def experiment_weight_decay():
    print("=" * 56)
    print(" 실험 B: AdamW의 weight decay 효과 (과적합 방지)")
    print("=" * 56)
    print("weight decay(wd)를 키우며 최종 가중치 크기를 관찰\n")

    print(f"{'wd':>8} {'최종 손실':>14} {'최종 |params|':>16}")
    print("-" * 40)
    results = {}
    for wd in [0.0, 0.01, 0.1]:
        hist, pnorm = adamw(lr=0.3, wd=wd)
        results[wd] = pnorm
        print(f"{wd:>8} {hist[-1]:>14.6f} {pnorm[-1]:>16.5f}")
    print()
    print("-> wd가 클수록 가중치가 0쪽으로 더 당겨진다(크기 작아짐).")
    print("   가중치가 작으면 모델이 단순해져 과적합이 줄어든다.")
    print("   LLM 파인튜닝에서 적당한 wd(보통 0.01)가 표준인 이유.\n")
    return results


# ----------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------

def plot_all(h_sgd, h_mom, h_adam, wd_results, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (좌) 옵티마이저별 수렴 속도 (로그 스케일이라 차이가 잘 보임)
    steps = range(len(h_sgd))
    ax1.plot(steps, h_sgd, label="SGD", linewidth=2, color="#ef4444")
    ax1.plot(steps, h_mom, label="Momentum", linewidth=2, color="#f59e0b")
    ax1.plot(steps, h_adam, label="Adam", linewidth=2, color="#2563eb")
    ax1.set_yscale("log")  # 손실 차이가 커서 로그 스케일
    ax1.set_title("Convergence Speed (log scale)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss (log)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # (우) AdamW weight decay별 가중치 크기 변화
    for wd, pnorm in wd_results.items():
        ax2.plot(range(len(pnorm)), pnorm, label=f"wd = {wd}", linewidth=2)
    ax2.set_title("AdamW: Weight Decay shrinks parameters", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("|params| (weight size)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"그래프 저장됨: {out_path}")


def main():
    print("\n" + "=" * 56)
    print(" Day 44: 옵티마이저 심화 — Adam & AdamW")
    print("=" * 56 + "\n")

    h_sgd, h_mom, h_adam = experiment_convergence()
    wd_results = experiment_weight_decay()

    plot_all(h_sgd, h_mom, h_adam, wd_results, "optimizer_comparison.png")

    print("=" * 56)
    print(" 정리")
    print("=" * 56)
    print("SGD      : 같은 학습률 -> 느리고 진동")
    print("Momentum : 관성으로 진동 억제 + 가속")
    print("Adam     : 파라미터별 학습률 자동 조정 (b1=0.9, b2=0.999)")
    print("AdamW    : Adam + 분리된 weight decay -> LLM 파인튜닝 표준")
    print("\n튜닝 포인트: learning rate(보폭), weight decay(과적합 억제 강도)")


if __name__ == "__main__":
    main()