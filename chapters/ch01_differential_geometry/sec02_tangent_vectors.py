"""
§1.2 接ベクトルと接空間 - パラメータ変化の方向を幾何学的に捉える

接ベクトルは「多様体上の点における方向」を表します。
統計的多様体では、パラメータの微小変化の方向に対応します。

【既知概念との対応】
- 接ベクトル → パラメータの微小変化（∂θ）
- 接空間 → その点で可能な全ての変化方向
- スコア関数 → 対数尤度の勾配（まさに接ベクトル的）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def gaussian_pdf(x, mu, sigma):
    """正規分布の確率密度関数"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# =============================================================================
# 1. 接ベクトルの直感的理解
# =============================================================================

def plot_tangent_vector_intuition():
    """
    接ベクトル = 多様体上の点における「方向」

    パラメータ空間の点 p = (μ₀, σ₀) における接ベクトルは、
    パラメータの微小変化 (dμ, dσ) に対応する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図：パラメータ空間での接ベクトル
    ax1 = axes[0]

    # 基準点
    mu0, sigma0 = 0, 1
    ax1.plot(mu0, sigma0, 'ko', markersize=12, label=f'p = ({mu0}, {sigma0})')

    # 接ベクトルの例
    tangent_vectors = [
        ((1, 0), 'r', '∂/∂μ方向: μを増やす'),
        ((0, 0.5), 'b', '∂/∂σ方向: σを増やす'),
        ((0.7, 0.7), 'g', '混合方向: μとσを同時に増やす'),
        ((-0.5, 0.3), 'm', '別の方向'),
    ]

    for (dmu, dsigma), color, label in tangent_vectors:
        ax1.annotate('', xy=(mu0 + dmu, sigma0 + dsigma), xytext=(mu0, sigma0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax1.text(mu0 + dmu * 1.1, sigma0 + dsigma * 1.1, label, fontsize=9, color=color)

    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('σ', fontsize=12)
    ax1.set_title('パラメータ空間における接ベクトル\n点pでの「方向」を表す', fontsize=11)
    ax1.set_xlim(-1.5, 2)
    ax1.set_ylim(0, 2.5)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 右図：分布空間での対応する変化
    ax2 = axes[1]
    x = np.linspace(-4, 4, 200)

    # 基準分布
    y0 = gaussian_pdf(x, mu0, sigma0)
    ax2.plot(x, y0, 'k-', linewidth=2, label=f'p(x|{mu0},{sigma0}) 基準')

    # 各方向への微小変化
    epsilon = 0.3
    changes = [
        ((mu0 + epsilon, sigma0), 'r', '--', f'μ+ε'),
        ((mu0, sigma0 + epsilon * 0.5), 'b', '--', f'σ+ε'),
        ((mu0 + epsilon * 0.7, sigma0 + epsilon * 0.7), 'g', '--', 'μ,σ+ε'),
    ]

    for (mu, sigma), color, style, label in changes:
        y = gaussian_pdf(x, mu, sigma)
        ax2.plot(x, y, color=color, linestyle=style, linewidth=1.5, label=label)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('p(x)', fontsize=12)
    ax2.set_title('対応する分布の変化\n接ベクトルが示す方向に分布が変化', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tangent_vectors_01_intuition.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. 接ベクトル = パラメータの微小変化の「方向」
2. 点 p = (μ, σ) での接空間は、可能な全ての方向の集合
3. 正規分布の場合、接空間は2次元（∂/∂μ と ∂/∂σ が基底）
4. 接ベクトルは確率分布の変化方向を指定する
""")


# =============================================================================
# 2. スコア関数と接ベクトルの関係
# =============================================================================

def plot_score_function():
    """
    スコア関数 = 対数尤度の勾配 → 接ベクトルの具体的表現

    正規分布のスコア関数：
    - ∂log p(x|μ,σ)/∂μ = (x - μ)/σ²
    - ∂log p(x|μ,σ)/∂σ = -1/σ + (x-μ)²/σ³

    これらは確率分布の微小変化を「関数として」表現したもの。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mu0, sigma0 = 0, 1
    x = np.linspace(-4, 4, 200)

    # 左上：確率密度関数
    ax1 = axes[0, 0]
    y = gaussian_pdf(x, mu0, sigma0)
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.fill_between(x, y, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    ax1.set_title(f'確率密度関数 N({mu0}, {sigma0}²)')
    ax1.grid(True, alpha=0.3)

    # 右上：対数尤度
    ax2 = axes[0, 1]
    log_p = np.log(y + 1e-10)  # 数値安定性
    ax2.plot(x, log_p, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('log p(x)')
    ax2.set_title('対数尤度 log p(x|μ,σ)')
    ax2.grid(True, alpha=0.3)

    # 左下：μに関するスコア関数
    ax3 = axes[1, 0]
    score_mu = (x - mu0) / sigma0**2
    ax3.plot(x, score_mu, 'r-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('∂log p/∂μ')
    ax3.set_title('スコア関数（μ方向）\n= (x-μ)/σ²')
    ax3.grid(True, alpha=0.3)

    # 右下：σに関するスコア関数
    ax4 = axes[1, 1]
    score_sigma = -1/sigma0 + (x - mu0)**2 / sigma0**3
    ax4.plot(x, score_sigma, 'm-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('∂log p/∂σ')
    ax4.set_title('スコア関数（σ方向）\n= -1/σ + (x-μ)²/σ³')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tangent_vectors_02_score_function.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. スコア関数 = 対数尤度のパラメータ微分
2. これは「接ベクトルを関数として表現」したもの
3. 統計的多様体では、接ベクトルを関数（確率変数）として扱う
4. Fisher情報行列 = スコア関数の共分散
   I(θ) = E[∂log p/∂θ · ∂log p/∂θᵀ]
5. これがリーマン計量になる（第2章で詳述）
""")


# =============================================================================
# 3. 接空間の基底と座標表示
# =============================================================================

def plot_tangent_space_basis():
    """
    接空間の基底ベクトルと任意の接ベクトルの表現

    点 p での接空間 TₚM は線形空間：
    - 基底: {∂/∂μ, ∂/∂σ}
    - 任意の接ベクトル: v = a(∂/∂μ) + b(∂/∂σ)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図：接空間の基底
    ax1 = axes[0]
    mu0, sigma0 = 0, 1

    # 多様体上の点
    ax1.plot(mu0, sigma0, 'ko', markersize=12)
    ax1.text(mu0 - 0.2, sigma0 - 0.15, 'p', fontsize=14, fontweight='bold')

    # 基底ベクトル
    ax1.annotate('', xy=(mu0 + 1, sigma0), xytext=(mu0, sigma0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax1.text(mu0 + 1.05, sigma0 + 0.05, '∂/∂μ', fontsize=12, color='red')

    ax1.annotate('', xy=(mu0, sigma0 + 0.8), xytext=(mu0, sigma0),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax1.text(mu0 + 0.05, sigma0 + 0.85, '∂/∂σ', fontsize=12, color='blue')

    # 接空間を示す領域（概念的）
    from matplotlib.patches import Rectangle
    rect = Rectangle((mu0 - 0.8, sigma0 - 0.3), 2.2, 1.4,
                      fill=True, facecolor='yellow', alpha=0.2,
                      edgecolor='orange', linestyle='--')
    ax1.add_patch(rect)
    ax1.text(mu0 + 0.3, sigma0 + 0.9, '接空間 TₚM', fontsize=11, color='orange')

    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('σ', fontsize=12)
    ax1.set_title('接空間の基底ベクトル\n{∂/∂μ, ∂/∂σ}', fontsize=11)
    ax1.set_xlim(-1.5, 2)
    ax1.set_ylim(0, 2.5)
    ax1.grid(True, alpha=0.3)

    # 右図：任意の接ベクトルの分解
    ax2 = axes[1]
    ax2.plot(mu0, sigma0, 'ko', markersize=12)

    # 任意の接ベクトル v = 0.6(∂/∂μ) + 0.8(∂/∂σ)
    a, b = 0.6, 0.8
    ax2.annotate('', xy=(mu0 + a, sigma0 + b), xytext=(mu0, sigma0),
                 arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax2.text(mu0 + a + 0.05, sigma0 + b + 0.05, f'v = {a}(∂/∂μ) + {b}(∂/∂σ)',
             fontsize=11, color='green')

    # 分解成分
    ax2.annotate('', xy=(mu0 + a, sigma0), xytext=(mu0, sigma0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5, linestyle='--'))
    ax2.annotate('', xy=(mu0 + a, sigma0 + b), xytext=(mu0 + a, sigma0),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='--'))

    ax2.text(mu0 + a/2, sigma0 - 0.1, f'{a}(∂/∂μ)', fontsize=10, color='red')
    ax2.text(mu0 + a + 0.05, sigma0 + b/2, f'{b}(∂/∂σ)', fontsize=10, color='blue')

    ax2.set_xlabel('μ', fontsize=12)
    ax2.set_ylabel('σ', fontsize=12)
    ax2.set_title('接ベクトルの座標表示\nv = a(∂/∂μ) + b(∂/∂σ)', fontsize=11)
    ax2.set_xlim(-1, 2)
    ax2.set_ylim(0, 2.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tangent_vectors_03_basis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. 接空間 TₚM は線形空間（ベクトル空間）
2. n次元多様体の接空間はn次元
3. 座標系 (θ¹, θ², ..., θⁿ) に対して基底 {∂/∂θⁱ}
4. 任意の接ベクトルは基底の線形結合で表せる
5. 座標変換すると基底も変換される（共変性）
""")


# =============================================================================
# 4. カルマンフィルタとの接点
# =============================================================================

def plot_kalman_connection():
    """
    カルマンフィルタの状態更新を接ベクトルの視点から見る

    カルマンフィルタでの状態更新:
    - 予測分布 N(μ_pred, σ²_pred) から観測を得て
    - 更新分布 N(μ_post, σ²_post) へ移動

    この「移動」は多様体上の曲線として捉えられる。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # パラメータ
    mu_pred, sigma_pred = 0, 2.0   # 事前分布
    mu_obs = 1.5                    # 観測値
    sigma_obs = 0.8                 # 観測ノイズ

    # カルマンゲインと更新
    K = sigma_pred**2 / (sigma_pred**2 + sigma_obs**2)
    mu_post = mu_pred + K * (mu_obs - mu_pred)
    sigma_post = np.sqrt((1 - K) * sigma_pred**2)

    # 左図：パラメータ空間での更新
    ax1 = axes[0]

    # 事前、事後、観測の点
    ax1.plot(mu_pred, sigma_pred, 'bo', markersize=12, label='事前分布')
    ax1.plot(mu_post, sigma_post, 'go', markersize=12, label='事後分布')
    ax1.plot(mu_obs, sigma_obs, 'r^', markersize=12, label='観測 (μ_obs, σ_obs)')

    # 更新の矢印（接ベクトル的）
    ax1.annotate('', xy=(mu_post, sigma_post), xytext=(mu_pred, sigma_pred),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax1.text((mu_pred + mu_post)/2 - 0.3, (sigma_pred + sigma_post)/2 + 0.1,
             '状態更新', fontsize=10, color='purple')

    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('σ', fontsize=12)
    ax1.set_title('カルマンフィルタの状態更新\nパラメータ空間（多様体）上の移動', fontsize=11)
    ax1.legend()
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3)

    # 右図：対応する分布の変化
    ax2 = axes[1]
    x = np.linspace(-5, 5, 200)

    y_pred = gaussian_pdf(x, mu_pred, sigma_pred)
    y_post = gaussian_pdf(x, mu_post, sigma_post)
    y_obs = gaussian_pdf(x, mu_obs, sigma_obs)

    ax2.plot(x, y_pred, 'b-', linewidth=2, label=f'事前 N({mu_pred:.1f}, {sigma_pred:.1f}²)')
    ax2.plot(x, y_post, 'g-', linewidth=2, label=f'事後 N({mu_post:.2f}, {sigma_post:.2f}²)')
    ax2.plot(x, y_obs, 'r--', linewidth=2, label=f'尤度 N({mu_obs:.1f}, {sigma_obs:.1f}²)')
    ax2.axvline(x=mu_obs, color='r', linestyle=':', alpha=0.5, label='観測値')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('p(x)', fontsize=12)
    ax2.set_title('対応する分布の変化', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tangent_vectors_04_kalman.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"""
【カルマンフィルタと情報幾何の接点】
- 事前: (μ, σ) = ({mu_pred:.1f}, {sigma_pred:.1f})
- 事後: (μ, σ) = ({mu_post:.2f}, {sigma_post:.2f})
- カルマンゲイン K = {K:.3f}

【学習ポイント】
1. カルマンフィルタの状態更新は多様体上の「移動」
2. 更新ベクトル (Δμ, Δσ) は接ベクトル的
3. 情報幾何では、この移動を「測地線」や「射影」で解釈できる
4. 自然勾配法もこの視点から理解できる（第4章で詳述）
""")


if __name__ == '__main__':
    print("=" * 60)
    print("§1.2 接ベクトルと接空間")
    print("=" * 60)

    print("\n[1/4] 接ベクトルの直感的理解")
    plot_tangent_vector_intuition()

    print("\n[2/4] スコア関数と接ベクトルの関係")
    plot_score_function()

    print("\n[3/4] 接空間の基底と座標表示")
    plot_tangent_space_basis()

    print("\n[4/4] カルマンフィルタとの接点")
    plot_kalman_connection()

    print("\n" + "=" * 60)
    print("次のステップ: sec03_metric_tensor.py でリーマン計量を学ぶ")
    print("=" * 60)
