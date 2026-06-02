# ================================================================
# Day 42-43: numpy로 선형회귀 완전 구현 (PyTorch/sklearn 금지)
#
# 전체 파이프라인을 처음부터 끝까지 직접 구현합니다:
#   데이터 생성 -> 모델 초기화 -> [순전파 -> 손실 -> 역전파 -> 업데이트] x 100
#
# 이 미니 루프가 곧 LLM 파인튜닝 루프의 축소판입니다.
# 구조가 완전히 동일: forward -> loss -> backward -> update
#
# 마지막에 직접 만든 결과를 scikit-learn과 비교해 검증합니다.
#
# 실행:  python3 linear_regression.py
# ================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서 파일로 저장하기 위한 백엔드
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------
# 1. 데이터 생성
# ----------------------------------------------------------------
# 진짜 관계: y = 2x + 1  + 약간의 노이즈(현실 데이터처럼)
# 노이즈가 있어야 '완벽한 직선'이 아니라 실제 학습 상황과 비슷해진다.

def make_data(n=100, true_w=2.0, true_b=1.0, noise=1.5, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n)              # 입력 0~10
    noise_vals = rng.normal(0, noise, size=n)   # 평균0, 표준편차 noise
    y = true_w * x + true_b + noise_vals        # 정답 = 진짜관계 + 노이즈
    return x, y


# ----------------------------------------------------------------
# 2. 입력 정규화 (표준화, Standardization)
# ----------------------------------------------------------------
# x' = (x - 평균) / 표준편차  -> 평균0, 표준편차1 로 변환.
#
# 왜 하나?
#   x의 평균이 5처럼 0에서 멀면, 절편 b가 수렴하는 속도가 매우 느려진다.
#   (w는 빨리 찾는데 b만 한참 뒤처지는 현상)
#   입력을 0 중심으로 맞추면 w, b가 비슷한 속도로 함께 수렴한다.
#   => 실무 딥러닝에서 입력 정규화는 거의 항상 하는 표준 전처리.

def normalize(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std


# ----------------------------------------------------------------
# 3. 모델 / 손실 / 기울기
# ----------------------------------------------------------------
# 모델(가설):  y_hat = w*x + b
# 손실(MSE):   L = (1/N) sum (y_hat - y)^2
# 기울기:
#     dL/dw = (2/N) sum (y_hat - y) * x
#     dL/db = (2/N) sum (y_hat - y)

def forward(x, w, b):
    """순전파: 입력 x로 예측값 계산"""
    return w * x + b

def mse_loss(y_hat, y):
    """손실: 평균제곱오차"""
    return np.mean((y_hat - y) ** 2)

def gradients(x, y, y_hat):
    """역전파: 손실을 w, b로 미분한 기울기"""
    n = len(x)
    error = y_hat - y
    dw = (2.0 / n) * np.sum(error * x)
    db = (2.0 / n) * np.sum(error)
    return dw, db


# ----------------------------------------------------------------
# 4. 학습 루프 (핵심 - LLM 파인튜닝과 구조 동일)
# ----------------------------------------------------------------

def train(x, y, lr=0.1, epochs=100):
    w, b = 0.0, 0.0          # 모델 초기화: 백지 상태에서 시작
    loss_history = []

    print(f"{'epoch':>6} {'loss':>12} {'w':>10} {'b':>10}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        y_hat = forward(x, w, b)         # 1) 순전파
        loss = mse_loss(y_hat, y)        # 2) 손실 계산
        dw, db = gradients(x, y, y_hat)  # 3) 역전파(기울기)
        w -= lr * dw                     # 4) 가중치 업데이트
        b -= lr * db

        loss_history.append(loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {loss:>12.6f} {w:>10.4f} {b:>10.4f}")

    return w, b, loss_history


# ----------------------------------------------------------------
# 5. 시각화: 학습 곡선 + 예측선
# ----------------------------------------------------------------

def plot_results(x, y, w, b, loss_history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (좌) 학습 곡선: epoch에 따른 loss 감소
    epochs = range(1, len(loss_history) + 1)
    ax1.plot(epochs, loss_history, color="#2563eb", linewidth=2)
    ax1.set_title("Learning Curve (Loss per Epoch)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f"final loss = {loss_history[-1]:.4f}",
                 xy=(len(loss_history), loss_history[-1]),
                 xytext=(len(loss_history) * 0.4, max(loss_history) * 0.5),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=10)

    # (우) 데이터 산점도 + 학습된 예측선 + 진짜 직선
    ax2.scatter(x, y, s=25, alpha=0.5, color="#94a3b8", label="data (with noise)")
    xs = np.linspace(x.min(), x.max(), 100)
    ax2.plot(xs, w * xs + b, color="#dc2626", linewidth=2.5,
             label=f"fitted: y = {w:.3f}x + {b:.3f}")
    ax2.plot(xs, 2.0 * xs + 1.0, color="#16a34a", linewidth=1.5,
             linestyle="--", label="true: y = 2x + 1")
    ax2.set_title("Fitted Line vs Data", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n그래프 저장됨: {out_path}")


# ----------------------------------------------------------------
# 6. scikit-learn과 비교 (검증용)
# ----------------------------------------------------------------

def compare_sklearn(x, y, my_w, my_b):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)  # sklearn은 2D 입력 요구
    sk_w = model.coef_[0]
    sk_b = model.intercept_

    print("\n" + "=" * 50)
    print(" scikit-learn 비교 검증")
    print("=" * 50)
    print(f"{'':>14}{'w (기울기)':>16}{'b (절편)':>14}")
    print("-" * 44)
    print(f"{'내 구현':>14}{my_w:>16.4f}{my_b:>14.4f}")
    print(f"{'scikit-learn':>14}{sk_w:>16.4f}{sk_b:>14.4f}")
    print(f"{'차이':>14}{abs(my_w - sk_w):>16.4f}{abs(my_b - sk_b):>14.4f}")
    print("-" * 44)
    if abs(my_w - sk_w) < 0.05 and abs(my_b - sk_b) < 0.05:
        print("=> 거의 완벽히 일치! 직접 만든 경사하강법이 정확히 동작함")
    else:
        print("=> 약간의 차이. epoch나 학습률을 조정해볼 수 있음")
    print("(sklearn은 정규방정식으로 '정확한' 최적해를 한 번에 계산,")
    print(" 내 구현은 경사하강법으로 '점진적으로' 도달 -> 둘이 일치하면 성공)")


# ----------------------------------------------------------------
# 메인
# ----------------------------------------------------------------

def main():
    print("=" * 50)
    print(" Day 42-43: numpy 선형회귀 완전 구현")
    print("=" * 50)
    print("진짜 관계: y = 2x + 1 (+ 노이즈)")
    print("목표: 데이터만 보고 w=2, b=1 을 스스로 찾기\n")

    # 1) 데이터
    x, y = make_data(n=100)

    # 2) 입력 정규화 (학습은 정규화된 공간에서)
    x_norm, x_mean, x_std = normalize(x)

    # 3) 학습
    w_n, b_n, loss_history = train(x_norm, y, lr=0.1, epochs=100)

    # 4) 정규화 공간의 (w,b)를 원래 x 공간으로 환산
    #    y = w_n*((x-mean)/std) + b_n = (w_n/std)*x + (b_n - w_n*mean/std)
    w = w_n / x_std
    b = b_n - w_n * x_mean / x_std

    print(f"\n학습 완료 (원래 공간): y = {w:.4f}x + {b:.4f}")
    print("(진짜 관계 y = 2x + 1 에 근접 — 노이즈 때문에 정확히 2,1은 아님)")

    # 5) 시각화 (원래 공간 기준)
    plot_results(x, y, w, b, loss_history,
                 "/mnt/user-data/outputs/learning_curve.png")

    # 6) sklearn 비교
    compare_sklearn(x, y, w, b)


if __name__ == "__main__":
    main()