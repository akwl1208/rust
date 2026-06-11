"""
================================================================
Day 63-65: Attention 메커니즘 직접 구현 (numpy scratch)
 
배우는 것:
  1. Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k)) V
  2. Q, K, V = 정보 검색 메타포 (질문/열쇠/값)
  3. Multi-Head Attention: 분할 -> 각각 attention -> 병합
  4. Causal Mask: 미래 토큰 차단
  5. RoPE: 회전으로 위치정보 주입 (맛보기)
 
실행:  python3 attention.py
================================================================
"""
import numpy as np
 
np.set_printoptions(precision=3, suppress=True)
 
 
def softmax(x):
    """마지막 축 기준 softmax. max를 빼는 건 overflow 방지(표준 트릭)."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# ================================================================
# 1) Scaled Dot-Product Attention — 가장 핵심
#    수식: softmax(Q K^T / sqrt(d_k)) V
# ================================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
 
    # 1단계: QK^T = 각 토큰쌍의 관련도 점수
    #   Q의 각 행(질문)을 K의 각 행(열쇠)과 내적 -> 점수표
    scores = Q @ K.swapaxes(-1, -2)
 
    # 2단계: sqrt(d_k)로 나눔 (scaling)
    #   d_k가 크면 점수가 커져 softmax가 한쪽으로 쏠림 -> 완화
    scores = scores / np.sqrt(d_k)
 
    # 3단계: (선택) causal mask — 미래 위치를 -inf로 막기
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
 
    # 4단계: softmax -> 각 행이 합=1인 확률(attention weights)
    weights = softmax(scores)
 
    # 5단계: weights @ V -> 점수만큼 V를 가중합한 새 표현
    output = weights @ V
    return output, weights

# ================================================================
# 2) Causal Mask 만들기
#    위치 (i, j)에서 j > i 면 미래 -> True(가림)
# ================================================================
def make_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

# ================================================================
# 3) Multi-Head Attention
#    d_model을 head로 쪼개 각각 attention 후 병합
# ================================================================
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    seq_len, d_model = X.shape
    d_k = d_model // n_heads
 
    # Q, K, V 만들기 (입력에 가중치 곱)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
 
    # 헤드 분할: (seq, d_model) -> (n_heads, seq, d_k)
    def split(M):
        return M.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
 
    Qh, Kh, Vh = split(Q), split(K), split(V)
 
    mask = make_causal_mask(seq_len) if causal else None
 
    # 헤드마다 attention
    outs = []
    for h in range(n_heads):
        o, _ = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask)
        outs.append(o)
 
    # 병합: 헤드들을 다시 이어붙임 -> (seq, d_model)
    concat = np.concatenate(outs, axis=-1)
 
    # 출력 프로젝션 (O projection) — LoRA를 자주 붙이는 그 O
    return concat @ Wo

# ================================================================
# 4) RoPE — 회전 위치 인코딩 (맛보기)
#    차원을 2개씩 짝지어, 위치(pos)에 비례해 회전시킨다.
#    같은 벡터라도 위치마다 다르게 변형 = 위치정보 주입
# ================================================================
def rope_rotate(vec, pos):
    d = len(vec)
    out = vec.copy()
    for i in range(0, d, 2):
        theta = pos / (10000 ** (i / d))
        c, s = np.cos(theta), np.sin(theta)
        out[i] = vec[i] * c - vec[i + 1] * s
        out[i + 1] = vec[i] * s + vec[i + 1] * c
    return out
 
 
if __name__ == "__main__":
    print("=== 1. Scaled Dot-Product Attention ===")
    np.random.seed(42)
    seq_len, d_k = 4, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
 
    out, w = scaled_dot_product_attention(Q, K, V)
    print("attention weights (행 합=1):")
    print(w)
    print("행 합 확인:", w.sum(axis=1))
    print("출력 shape:", out.shape)
 
    print("\n=== 2. Causal Mask 버전 ===")
    mask = make_causal_mask(seq_len)
    out_c, w_c = scaled_dot_product_attention(Q, K, V, mask)
    print("causal weights (위 삼각형=0, 미래 차단):")
    print(w_c)
 
    print("\n=== 3. Multi-Head Attention ===")
    np.random.seed(0)
    d_model, n_heads = 8, 2
    X = np.random.randn(seq_len, d_model)
    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)
    Wo = np.random.randn(d_model, d_model)
    mha = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=True)
    print("MHA 출력 shape:", mha.shape, "(입력과 같음)")
 
    print("\n=== 4. RoPE ===")
    v = np.array([1.0, 0.0, 1.0, 0.0])
    for pos in [0, 1, 2]:
        print(f"위치 {pos}: {rope_rotate(v, pos)}")
    print("-> 같은 벡터라도 위치마다 다르게 회전 = 위치정보 주입")
 