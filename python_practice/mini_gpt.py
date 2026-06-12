"""
=============================================================================
 Day 66-67  미니 GPT 직접 조립하기 (Python / numpy 버전)
=============================================================================

[오늘의 학습 목표]
  "블록을 직접 조립해봐야 '어디를 수정하면 어떤 효과가 나는가'를 알 수 있다."

그래서 이 파일은 라이브러리에 숨겨진 게 하나도 없습니다.
RMSNorm, SwiGLU, Attention, Transformer 블록을 전부 손으로 만듭니다.
numpy만 씁니다 (pip install numpy).

[전체 구조 한눈에 보기]
  토큰 입력 (정수)
      │
      ▼
  [Embedding]          : 정수 토큰 → 128차원 벡터
      │
      ▼
  [Transformer 블록] × 2   ← 오늘의 주인공. 아래 4개로 구성:
      │  ├─ RMSNorm  (정규화)
      │  ├─ Attention (토큰끼리 정보 교환)
      │  ├─ RMSNorm
      │  └─ SwiGLU FFN (각 토큰을 더 똑똑하게 변환)
      ▼
  [최종 RMSNorm]
      │
      ▼
  [lm_head]            : 128차원 → vocab 크기 (다음 토큰 확률용 logits)
=============================================================================
"""

import numpy as np

# 재현성을 위해 난수 고정 (매번 같은 결과가 나오게)
rng = np.random.default_rng(42)

# =============================================================================
# 1) RMSNorm  —  "정규화" 레이어
# =============================================================================
# 정규화란? 벡터 값들의 크기를 일정하게 맞춰주는 것.
# 학습이 안정적으로 되게 도와줍니다.
#
# LayerNorm vs RMSNorm 차이:
#   LayerNorm = (x - 평균) / 표준편차    ← 평균을 빼는 단계가 있음
#   RMSNorm   =  x        / RMS         ← 평균 빼기를 생략! 그래서 더 빠름
#
# RMS = Root Mean Square = sqrt(제곱들의 평균)
# 요즘 LLM(LLaMA, Mistral 등)은 거의 다 RMSNorm을 씁니다.
# =============================================================================
class RMSNorm:
    def __init__(self, d_model, eps=1e-6):
        # gain(g): 학습되는 파라미터. 정규화 후 각 차원을 얼마나 키울지 조절.
        # 처음엔 1로 시작 (즉, 정규화만 하고 안 건드림)
        self.gain = np.ones(d_model, dtype=np.float32)
        self.eps = eps  # 0으로 나누는 걸 막는 아주 작은 수

    def forward(self, x):
        # x 모양: (batch, seq, d_model)
        # 마지막 차원(d_model)에 대해서만 RMS를 계산합니다.
        #
        # 1) 각 값을 제곱하고 평균낸다
        mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)  # (batch, seq, 1)
        # 2) 루트 씌우면 RMS
        rms = np.sqrt(mean_sq + self.eps)                  # (batch, seq, 1)
        # 3) x를 RMS로 나눔 → 크기가 일정해짐 (브로드캐스트로 자동 정렬)
        normed = x / rms
        # 4) 학습된 gain을 곱함
        return normed * self.gain

# =============================================================================
# 2) SwiGLU FFN  —  "각 토큰을 똑똑하게 변환하는" 레이어
# =============================================================================
# FFN(Feed-Forward Network)은 토큰 하나하나를 독립적으로 변환합니다.
# (Attention이 '토큰끼리 섞는' 거라면, FFN은 '각자 생각하는' 단계)
#
# 보통 FFN: Linear → 활성화함수(ReLU 등) → Linear  (가중치 2개)
# SwiGLU  : 가중치를 3개 사용하고 '게이팅(gating)'을 추가 → 성능이 더 좋음
#
# 수식: SwiGLU(x) = ( SiLU(x@W_gate) * (x@W_up) ) @ W_down
#   - SiLU(z) = z * sigmoid(z)   ← Swish라고도 불리는 부드러운 활성화함수
#   - '*'는 원소별 곱(element-wise). gate가 up을 얼마나 통과시킬지 조절(게이팅)
# =============================================================================
def silu(z):
    # SiLU = z * sigmoid(z).  ReLU와 비슷하지만 부드럽게 꺾임.
    return z * (1.0 / (1.0 + np.exp(-z)))


class SwiGLUFFN:
    def __init__(self, d_model, d_ff):
        # d_ff는 보통 d_model보다 큽니다 (중간에서 넓게 펼쳤다가 다시 줄임)
        # 0.02를 곱하는 건 초기 가중치를 작게 만드는 표준 관행 (학습 안정화)
        self.W_gate = rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.02
        self.W_up   = rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.02
        self.W_down = rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.02

    def forward(self, x):
        # x 모양: (batch, seq, d_model)
        gate = silu(x @ self.W_gate)  # (batch, seq, d_ff)  ← 게이트 신호
        up   = x @ self.W_up          # (batch, seq, d_ff)  ← 값 신호
        fused = gate * up             # (batch, seq, d_ff)  ← 게이팅(원소별 곱)
        return fused @ self.W_down    # (batch, seq, d_model) ← 다시 원래 크기로

# =============================================================================
# 3) Multi-Head Attention  —  "토큰끼리 정보를 교환하는" 레이어
# =============================================================================
# Attention의 핵심 아이디어:
#   각 토큰이 "다른 어떤 토큰을 얼마나 주목(attend)할지"를 계산해서
#   그만큼 정보를 가져옵니다.
#
# Multi-Head = 이 주목 과정을 여러 개(head)로 나눠서 병렬로 수행.
#   head마다 다른 관점에서 토큰 관계를 봅니다.
#
# 여기서는 '인과(causal) 마스크'를 씁니다 = 미래 토큰은 못 보게 막음.
# (GPT는 다음 단어를 예측하므로, 정답을 미리 보면 안 되니까)
# =============================================================================
def softmax(x, axis=-1):
    # 점수들을 확률(합이 1)로 바꿔줌. max를 빼는 건 오버플로 방지용 표준 기법.
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model은 n_heads로 나누어떨어져야 함"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # head 하나가 담당하는 차원

        # Q(쿼리), K(키), V(밸류), O(출력) 4개의 가중치
        self.W_q = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.W_k = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.W_v = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.W_o = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02

    def forward(self, x):
        batch, seq, _ = x.shape

        # 1) 입력을 Q, K, V로 각각 변환
        q = x @ self.W_q  # (batch, seq, d_model)
        k = x @ self.W_k
        v = x @ self.W_v

        # 2) head 개수만큼 쪼갬: (batch, seq, d_model) → (batch, n_heads, seq, d_head)
        def split_heads(t):
            t = t.reshape(batch, seq, self.n_heads, self.d_head)
            return t.transpose(0, 2, 1, 3)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # 3) 주목 점수 = Q와 K의 내적 / sqrt(d_head)
        #    (sqrt로 나누는 건 점수가 너무 커지지 않게 하는 스케일링)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        # scores 모양: (batch, n_heads, seq, seq)  ← 토큰i가 토큰j를 보는 점수

        # 4) 인과 마스크: 미래(오른쪽 위 삼각형)를 -무한대로 막아서 softmax 후 0이 되게
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)  # 위쪽 삼각형 True
        scores = np.where(mask, -1e9, scores)

        # 5) softmax로 확률화 → V를 가중합
        attn = softmax(scores, axis=-1)  # (batch, n_heads, seq, seq)
        out = attn @ v                   # (batch, n_heads, seq, d_head)

        # 6) head들을 다시 합침: (batch, n_heads, seq, d_head) → (batch, seq, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        # 7) 출력 가중치로 한 번 더 변환
        return out @ self.W_o
    
    # =============================================================================
# 3) Multi-Head Attention  —  "토큰끼리 정보를 교환하는" 레이어
# =============================================================================
# Attention의 핵심 아이디어:
#   각 토큰이 "다른 어떤 토큰을 얼마나 주목(attend)할지"를 계산해서
#   그만큼 정보를 가져옵니다.
#
# Multi-Head = 이 주목 과정을 여러 개(head)로 나눠서 병렬로 수행.
#   head마다 다른 관점에서 토큰 관계를 봅니다.
#
# 여기서는 '인과(causal) 마스크'를 씁니다 = 미래 토큰은 못 보게 막음.
# (GPT는 다음 단어를 예측하므로, 정답을 미리 보면 안 되니까)
# =============================================================================
def softmax(x, axis=-1):
    # 점수들을 확률(합이 1)로 바꿔줌. max를 빼는 건 오버플로 방지용 표준 기법.
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model은 n_heads로 나누어떨어져야 함"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # head 하나가 담당하는 차원

        # Q(쿼리), K(키), V(밸류), O(출력) 4개의 가중치
        self.W_q = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.W_k = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.W_v = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.W_o = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02

    def forward(self, x):
        batch, seq, _ = x.shape

        # 1) 입력을 Q, K, V로 각각 변환
        q = x @ self.W_q  # (batch, seq, d_model)
        k = x @ self.W_k
        v = x @ self.W_v

        # 2) head 개수만큼 쪼갬: (batch, seq, d_model) → (batch, n_heads, seq, d_head)
        def split_heads(t):
            t = t.reshape(batch, seq, self.n_heads, self.d_head)
            return t.transpose(0, 2, 1, 3)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # 3) 주목 점수 = Q와 K의 내적 / sqrt(d_head)
        #    (sqrt로 나누는 건 점수가 너무 커지지 않게 하는 스케일링)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        # scores 모양: (batch, n_heads, seq, seq)  ← 토큰i가 토큰j를 보는 점수

        # 4) 인과 마스크: 미래(오른쪽 위 삼각형)를 -무한대로 막아서 softmax 후 0이 되게
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)  # 위쪽 삼각형 True
        scores = np.where(mask, -1e9, scores)

        # 5) softmax로 확률화 → V를 가중합
        attn = softmax(scores, axis=-1)  # (batch, n_heads, seq, seq)
        out = attn @ v                   # (batch, n_heads, seq, d_head)

        # 6) head들을 다시 합침: (batch, n_heads, seq, d_head) → (batch, seq, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        # 7) 출력 가중치로 한 번 더 변환
        return out @ self.W_o
    
# =============================================================================
# 4) Transformer 블록  —  오늘의 핵심! 위 조각들을 조립
# =============================================================================
# 'pre-norm' 구조를 씁니다 = 정규화를 먼저 하고 그 결과를 레이어에 넣는 방식.
# (요즘 LLM 표준. 학습이 더 안정적입니다.)
#
# 잔차 연결(residual, "x + ..."):
#   레이어를 통과한 결과를 원래 입력에 '더합니다'.
#   이러면 정보가 사라지지 않고, 깊게 쌓아도 학습이 잘 됩니다.
#
#   x = x + Attention(RMSNorm(x))    ← 1단계: 토큰끼리 정보 교환
#   x = x + SwiGLU   (RMSNorm(x))    ← 2단계: 각 토큰 변환
#
# ★ 학습목표 연결: LoRA를 붙인다면 보통 Attention의 W_q, W_v에 붙입니다.
#   이 블록 구조가 눈에 보여야 "어디에 붙일지"가 보입니다.
# =============================================================================
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = RMSNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model, d_ff)

    def forward(self, x):
        # 1단계: pre-norm → attention → 잔차 더하기
        x = x + self.attn.forward(self.norm1.forward(x))
        # 2단계: pre-norm → FFN → 잔차 더하기
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

# =============================================================================
# 5) 미니 GPT  —  블록을 N개 쌓아서 완성
# =============================================================================
class MiniGPT:
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256):
        # 설정값을 저장 (나중에 파라미터 수 계산에 사용)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        # Embedding: 정수 토큰 ID → d_model 차원 벡터로 바꾸는 '사전(표)'
        # 모양 (vocab_size, d_model). 토큰 ID로 해당 행을 골라옵니다.
        self.token_emb = rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02

        # Transformer 블록을 n_layers개 쌓기
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # 마지막 정규화
        self.final_norm = RMSNorm(d_model)

        # lm_head: d_model → vocab_size.  각 토큰 위치에서 다음 토큰 점수(logits) 출력
        self.lm_head = rng.standard_normal((d_model, vocab_size)).astype(np.float32) * 0.02

    def forward(self, tokens):
        # tokens 모양: (batch, seq), 값은 0~vocab_size-1 사이 정수
        # 1) 임베딩 조회: 표에서 해당 토큰 행을 가져옴
        x = self.token_emb[tokens]  # (batch, seq, d_model)

        # 2) 블록들을 차례로 통과
        for block in self.blocks:
            x = block.forward(x)

        # 3) 최종 정규화
        x = self.final_norm.forward(x)

        # 4) logits 계산
        logits = x @ self.lm_head  # (batch, seq, vocab_size)
        return logits

    def num_params(self):
        """파라미터 수를 직접 세어봅니다 (모델 크기 이해)."""
        total = 0
        # 임베딩
        total += self.token_emb.size
        # 블록들
        for _ in range(self.n_layers):
            # RMSNorm 2개 (gain만)
            total += self.d_model * 2
            # Attention: W_q, W_k, W_v, W_o (각각 d_model x d_model)
            total += 4 * self.d_model * self.d_model
            # SwiGLU: W_gate, W_up (d_model x d_ff), W_down (d_ff x d_model)
            total += 2 * self.d_model * self.d_ff
            total += self.d_ff * self.d_model
        # 최종 norm
        total += self.d_model
        # lm_head
        total += self.d_model * self.vocab_size
        return total

# =============================================================================
# 실행해보기
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Day 66-67  미니 GPT 조립 (Python/numpy)")
    print("=" * 60)

    # [d_model=128, n_heads=4, n_layers=2] 커리큘럼 지정 설정
    model = MiniGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=256)

    # 더미 입력: batch=2개 문장, 각 seq=16개 토큰
    # 0~999 사이 정수를 랜덤으로 채움
    tokens = rng.integers(0, 1000, size=(2, 16))

    logits = model.forward(tokens)

    print(f"\n입력 토큰 모양 : {tokens.shape}      (batch=2, seq=16)")
    print(f"출력 logits 모양: {logits.shape}  (batch=2, seq=16, vocab=1000)")
    print(f"\n총 파라미터 수 : {model.num_params():,} 개")

    # 모델 크기 이해: 어디에 파라미터가 몰려있나?
    emb = model.token_emb.size
    head = model.d_model * model.vocab_size
    print(f"  - 임베딩    : {emb:,} 개")
    print(f"  - lm_head   : {head:,} 개")
    print(f"  - 위 둘 합  : {emb + head:,} 개  (전체의 큰 비중!)")
    print("\n  → 작은 모델에선 vocab 관련 파라미터가 절반 가까이를 차지합니다.")
    print("    d_model, n_layers를 키워야 Transformer 본체가 커집니다.")