"""
§1.5 リーマン計量 - Fisher情報行列が計量になる

リーマン計量は多様体上で「距離」や「角度」を測る道具です。
統計的多様体では、Fisher情報行列がリーマン計量の役割を果たします。

【既知概念との対応】
- Fisher情報行列 → リーマン計量テンソル
- KLダイバージェンス（2次近似） → リーマン距離の2乗
- スコア関数の共分散 → 計量の定義
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def gaussian_pdf(x, mu, sigma):
    """正規分布の確率密度関数"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# =============================================================================
# 1. Fisher情報行列の計算と意味
# =============================================================================

def compute_fisher_information_gaussian(mu, sigma):
    """
    正規分布 N(μ, σ²) のFisher情報行列を計算

    I(μ, σ) = E[∂log p/∂θ · ∂log p/∂θᵀ]

    正規分布の場合：
    I = [[1/σ², 0    ],
         [0,    2/σ²]]
    """
    I = np.array([
        [1 / sigma**2, 0],
        [0, 2 / sigma**2]
    ])
    return I


def plot_fisher_information():
    """
    Fisher情報行列の各成分の意味を可視化

    I_μμ = 1/σ²  : μに関する情報量（σが小さいほど大）
    I_σσ = 2/σ²  : σに関する情報量
    I_μσ = 0     : μとσは統計的に独立
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sigma_range = np.linspace(0.3, 3, 100)

    # I_μμ = 1/σ²
    ax1 = axes[0]
    I_mu_mu = 1 / sigma_range**2
    ax1.plot(sigma_range, I_mu_mu, 'r-', linewidth=2)
    ax1.set_xlabel('σ', fontsize=12)
    ax1.set_ylabel('I_μμ = 1/σ²', fontsize=12)
    ax1.set_title('μに関するFisher情報量\nσが小さいほど情報が多い', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # I_σσ = 2/σ²
    ax2 = axes[1]
    I_sigma_sigma = 2 / sigma_range**2
    ax2.plot(sigma_range, I_sigma_sigma, 'b-', linewidth=2)
    ax2.set_xlabel('σ', fontsize=12)
    ax2.set_ylabel('I_σσ = 2/σ²', fontsize=12)
    ax2.set_title('σに関するFisher情報量', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 情報行列の楕円表示
    ax3 = axes[2]
    points = [(0, 0.5), (0, 1.0), (0, 2.0)]

    for mu, sigma in points:
        I = compute_fisher_information_gaussian(mu, sigma)

        # 楕円のサイズはFisher情報の逆数に比例（Cramér-Raoの意味で）
        width = 2 * sigma  # ∝ 1/√I_μμ
        height = np.sqrt(2) * sigma  # ∝ 1/√I_σσ

        ellipse = Ellipse((mu, sigma), width=width * 0.3, height=height * 0.3,
                          fill=False, edgecolor='purple', linewidth=2)
        ax3.add_patch(ellipse)
        ax3.plot(mu, sigma, 'ko', markersize=6)

    ax3.set_xlabel('μ', fontsize=12)
    ax3.set_ylabel('σ', fontsize=12)
    ax3.set_title('Fisher情報の楕円表示\n小さいσで楕円が小さい（情報が多い）', fontsize=11)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(0, 3)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metric_tensor_01_fisher.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. Fisher情報行列 I(θ) = E[∂log p/∂θ · ∂log p/∂θᵀ]
2. 正規分布では I = diag(1/σ², 2/σ²)
3. σが小さい → Fisher情報が大きい → 推定精度が高い
4. Cramér-Raoの下界: Var(θ̂) ≥ I(θ)⁻¹
5. 楕円が小さい ⇔ 推定誤差の下界が小さい
""")


# =============================================================================
# 2. リーマン計量としてのFisher情報
# =============================================================================

def plot_riemannian_metric():
    """
    Fisher情報行列がリーマン計量になることを可視化

    リーマン計量の役割：
    - 接ベクトルの「長さ」を測る: ||v||² = vᵀ I v
    - 接ベクトル間の「角度」を測る: cos θ = <u,v> / (||u|| ||v||)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図：ユークリッド計量 vs Fisher計量
    ax1 = axes[0]
    mu0, sigma0 = 0, 1
    I = compute_fisher_information_gaussian(mu0, sigma0)

    # 単位円（ユークリッド）
    theta = np.linspace(0, 2*np.pi, 100)
    x_euclid = np.cos(theta)
    y_euclid = np.sin(theta)
    ax1.plot(x_euclid * 0.5 + mu0, y_euclid * 0.5 + sigma0, 'b--',
             linewidth=2, label='ユークリッド単位円')

    # Fisher計量による「単位円」（楕円になる）
    # ||v||²_Fisher = v1²/σ² + v2² · 2/σ² = 1
    # v1²/σ² + 2v2²/σ² = 1
    # v1² + 2v2² = σ²
    x_fisher = sigma0 * np.cos(theta)
    y_fisher = sigma0 / np.sqrt(2) * np.sin(theta)
    ax1.plot(x_fisher * 0.5 + mu0, y_fisher * 0.5 + sigma0, 'r-',
             linewidth=2, label='Fisher単位「円」')

    ax1.plot(mu0, sigma0, 'ko', markersize=10)
    ax1.text(mu0 - 0.1, sigma0 - 0.15, 'p', fontsize=12, fontweight='bold')

    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('σ', fontsize=12)
    ax1.set_title('ユークリッド計量 vs Fisher計量\n「単位球」の形が違う', fontsize=11)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 右図：異なる点での計量楕円
    ax2 = axes[1]

    # 複数の点でFisher計量楕円を描画
    points = [
        (0, 0.5, 'red'),
        (1, 1.0, 'blue'),
        (-1, 1.5, 'green'),
        (0, 2.0, 'purple'),
    ]

    for mu, sigma, color in points:
        # 単位楕円（Fisher計量で長さ1）
        x = sigma * np.cos(theta) * 0.2  # スケール調整
        y = sigma / np.sqrt(2) * np.sin(theta) * 0.2
        ax2.plot(x + mu, y + sigma, color=color, linewidth=1.5)
        ax2.plot(mu, sigma, 'o', color=color, markersize=8)

    ax2.set_xlabel('μ', fontsize=12)
    ax2.set_ylabel('σ', fontsize=12)
    ax2.set_title('各点でのFisher計量楕円\nσが小さいほど楕円が小さい（距離が大きく感じる）', fontsize=11)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(0, 3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metric_tensor_02_riemannian.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. リーマン計量 g(v, w) は各点で定義される内積
2. Fisher情報行列 I(θ) がこの役割を果たす
3. ||v||²_Fisher = vᵀ I(θ) v
4. σが小さい領域では「距離が大きく感じる」
   - 狭い分布を少し変えると大きな情報変化
5. これが「情報幾何」の名前の由来
""")


# =============================================================================
# 3. KLダイバージェンスとの関係
# =============================================================================

def plot_kl_and_metric():
    """
    KLダイバージェンスの2次近似がリーマン距離の2乗になる

    D_KL(p_θ || p_{θ+dθ}) ≈ (1/2) dθᵀ I(θ) dθ

    これが「Fisher情報 = リーマン計量」の根拠。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図：KLダイバージェンスの等高線
    ax1 = axes[0]
    mu0, sigma0 = 0, 1

    mu_range = np.linspace(-1, 1, 100)
    sigma_range = np.linspace(0.5, 1.5, 100)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)

    # KLダイバージェンス D_KL(N(μ₀,σ₀²) || N(μ,σ²))
    def kl_gaussian(mu1, s1, mu2, s2):
        return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5

    KL = kl_gaussian(mu0, sigma0, MU, SIGMA)

    contour = ax1.contour(MU, SIGMA, KL, levels=[0.01, 0.05, 0.1, 0.2, 0.5],
                          colors='blue')
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(mu0, sigma0, 'r*', markersize=15, label='基準点')
    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('σ', fontsize=12)
    ax1.set_title('KLダイバージェンスの等高線\nD_KL(基準 || 他)', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右図：2次近似（Fisher計量による）
    ax2 = axes[1]

    # Fisher計量による2次形式: (1/2)(Δμ²/σ² + 2Δσ²/σ²)
    dMU = MU - mu0
    dSIGMA = SIGMA - sigma0
    quad_approx = 0.5 * (dMU**2 / sigma0**2 + 2 * dSIGMA**2 / sigma0**2)

    contour2 = ax2.contour(MU, SIGMA, quad_approx, levels=[0.01, 0.05, 0.1, 0.2, 0.5],
                           colors='red')
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.plot(mu0, sigma0, 'r*', markersize=15, label='基準点')
    ax2.set_xlabel('μ', fontsize=12)
    ax2.set_ylabel('σ', fontsize=12)
    ax2.set_title('2次近似（Fisher計量）\n(1/2)dθᵀ I dθ', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metric_tensor_03_kl_metric.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. KLダイバージェンスは非対称: D_KL(p||q) ≠ D_KL(q||p)
2. しかし2次の近似で対称になる:
   D_KL(p_θ || p_{θ+dθ}) ≈ (1/2) dθᵀ I(θ) dθ
3. この2次形式がリーマン距離の2乗
4. 等高線の形がFisher計量楕円に対応
5. あなたのKLダイバージェンスの経験が直接活きる！
""")


# =============================================================================
# 4. 計量による距離の計算例
# =============================================================================

def plot_distance_comparison():
    """
    ユークリッド距離 vs Fisher距離の違いを可視化

    同じ「座標の差」でも、Fisher距離は σ に依存して変わる。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図：同じ座標距離でもFisher距離は異なる
    ax1 = axes[0]

    # 2つの経路：どちらも座標距離は同じ
    # 経路1: σが小さい領域 (0.5 → 0.6)
    # 経路2: σが大きい領域 (2.0 → 2.1)

    # 経路1
    mu1, sigma1_start, sigma1_end = 0, 0.5, 0.6
    ax1.plot([mu1, mu1], [sigma1_start, sigma1_end], 'r-', linewidth=3,
             label=f'経路1: σ={sigma1_start}→{sigma1_end}')
    ax1.plot(mu1, sigma1_start, 'ro', markersize=10)
    ax1.plot(mu1, sigma1_end, 'rs', markersize=10)

    # 経路2
    mu2, sigma2_start, sigma2_end = 1, 2.0, 2.1
    ax1.plot([mu2, mu2], [sigma2_start, sigma2_end], 'b-', linewidth=3,
             label=f'経路2: σ={sigma2_start}→{sigma2_end}')
    ax1.plot(mu2, sigma2_start, 'bo', markersize=10)
    ax1.plot(mu2, sigma2_end, 'bs', markersize=10)

    # Fisher距離の計算
    # ds² = (dμ)²/σ² + 2(dσ)²/σ²
    # σ方向のみの移動: ds = √2 |dσ|/σ
    d_sigma = 0.1
    fisher_dist_1 = np.sqrt(2) * d_sigma / sigma1_start
    fisher_dist_2 = np.sqrt(2) * d_sigma / sigma2_start

    ax1.text(mu1 + 0.1, (sigma1_start + sigma1_end)/2,
             f'Fisher距離≈{fisher_dist_1:.3f}', fontsize=10, color='red')
    ax1.text(mu2 + 0.1, (sigma2_start + sigma2_end)/2,
             f'Fisher距離≈{fisher_dist_2:.3f}', fontsize=10, color='blue')

    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('σ', fontsize=12)
    ax1.set_title('同じ座標距離でもFisher距離は異なる\nσが小さい領域は「遠く感じる」', fontsize=11)
    ax1.legend()
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3)

    # 右図：対応する分布の変化
    ax2 = axes[1]
    x = np.linspace(-4, 6, 200)

    # 経路1の分布変化
    y1_start = gaussian_pdf(x, mu1, sigma1_start)
    y1_end = gaussian_pdf(x, mu1, sigma1_end)
    ax2.plot(x, y1_start, 'r-', linewidth=2, label=f'σ={sigma1_start}')
    ax2.plot(x, y1_end, 'r--', linewidth=2, label=f'σ={sigma1_end}')

    # 経路2の分布変化
    y2_start = gaussian_pdf(x, mu2, sigma2_start)
    y2_end = gaussian_pdf(x, mu2, sigma2_end)
    ax2.plot(x, y2_start, 'b-', linewidth=2, label=f'σ={sigma2_start}')
    ax2.plot(x, y2_end, 'b--', linewidth=2, label=f'σ={sigma2_end}')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('p(x)', fontsize=12)
    ax2.set_title('対応する分布の変化\n狭い分布(赤)の方が変化が大きく見える', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('metric_tensor_04_distance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"""
【計算結果】
- 経路1 (σ=0.5→0.6): Fisher距離 ≈ {fisher_dist_1:.4f}
- 経路2 (σ=2.0→2.1): Fisher距離 ≈ {fisher_dist_2:.4f}
- 比率: {fisher_dist_1/fisher_dist_2:.2f}倍

【学習ポイント】
1. 座標の変化量が同じでも、Fisher距離は違う
2. σが小さい領域での変化は「情報的に大きい」
3. これがカルマンフィルタで観測精度が高いほど更新が効く理由の幾何学的説明
4. 自然勾配法では、この計量を考慮して最適化する
""")


if __name__ == '__main__':
    print("=" * 60)
    print("§1.5 リーマン計量 - Fisher情報行列が計量になる")
    print("=" * 60)

    print("\n[1/4] Fisher情報行列の計算と意味")
    plot_fisher_information()

    print("\n[2/4] リーマン計量としてのFisher情報")
    plot_riemannian_metric()

    print("\n[3/4] KLダイバージェンスとの関係")
    plot_kl_and_metric()

    print("\n[4/4] 計量による距離の計算例")
    plot_distance_comparison()

    print("\n" + "=" * 60)
    print("次のステップ: sec04_geodesics.py で測地線を学ぶ")
    print("=" * 60)
