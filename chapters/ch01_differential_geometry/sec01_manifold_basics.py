"""
§1.1 微分可能多様体 - 正規分布を例に理解する

情報幾何における「多様体」の最も身近な例は、確率分布のパラメータ空間です。
正規分布 N(μ, σ²) のパラメータ空間は2次元多様体を形成します。

【既知概念との対応】
- パラメータ空間 → 多様体
- パラメータの組 (μ, σ) → 多様体上の点
- パラメータの微小変化 → 接ベクトル
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1. 正規分布のパラメータ空間を多様体として可視化
# =============================================================================

def gaussian_pdf(x, mu, sigma):
    """正規分布の確率密度関数"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_parameter_space_as_manifold():
    """
    正規分布のパラメータ空間を2D平面として可視化

    重要な理解：
    - 各点 (μ, σ) は一つの正規分布に対応
    - この平面全体が「統計的多様体」
    - σ > 0 の制約があるため、上半平面のみ（境界を含まない開集合）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図：パラメータ空間（多様体）
    ax1 = axes[0]
    mu_range = np.linspace(-3, 3, 20)
    sigma_range = np.linspace(0.1, 3, 20)

    # グリッド点をプロット
    for mu in mu_range[::2]:
        for sigma in sigma_range[::2]:
            ax1.plot(mu, sigma, 'b.', markersize=3, alpha=0.5)

    # いくつかの特定の分布を強調
    special_points = [
        (0, 1, 'r', '標準正規分布'),
        (1, 0.5, 'g', '狭い分布'),
        (-1, 2, 'm', '広い分布'),
    ]

    for mu, sigma, color, label in special_points:
        ax1.plot(mu, sigma, 'o', color=color, markersize=10, label=label)

    ax1.set_xlabel('μ (平均)', fontsize=12)
    ax1.set_ylabel('σ (標準偏差)', fontsize=12)
    ax1.set_title('パラメータ空間 = 統計的多様体\n各点が一つの確率分布に対応', fontsize=12)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='σ=0 は含まない')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-0.5, 4)
    ax1.grid(True, alpha=0.3)

    # 右図：対応する確率密度関数
    ax2 = axes[1]
    x = np.linspace(-6, 6, 200)

    for mu, sigma, color, label in special_points:
        y = gaussian_pdf(x, mu, sigma)
        ax2.plot(x, y, color=color, linewidth=2, label=label)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('p(x)', fontsize=12)
    ax2.set_title('各点に対応する確率密度関数', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('manifold_basics_01_parameter_space.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. 多様体とは：パラメータの組が張る空間
2. 正規分布の場合：(μ, σ) の2次元空間（σ > 0）
3. 多様体上の各点は、一つの確率分布に対応
4. この視点が情報幾何の出発点
""")


# =============================================================================
# 2. 局所座標系の概念
# =============================================================================

def plot_local_coordinates():
    """
    多様体における局所座標系の概念を可視化

    同じ点を異なるパラメータ化で表現できる：
    - 座標系1: (μ, σ)    - 自然な座標
    - 座標系2: (μ, σ²)   - 分散を使う座標
    - 座標系3: (μ, 1/σ²) - 精度を使う座標（ベイズ推定で頻出）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 同じ分布族を異なる座標系で表示
    mu_vals = np.linspace(-2, 2, 10)
    sigma_vals = np.linspace(0.5, 2, 10)

    # 座標系1: (μ, σ)
    ax1 = axes[0]
    for mu in mu_vals:
        for sigma in sigma_vals:
            ax1.plot(mu, sigma, 'b.', markersize=4)
    ax1.set_xlabel('μ')
    ax1.set_ylabel('σ')
    ax1.set_title('座標系1: (μ, σ)')
    ax1.grid(True, alpha=0.3)

    # 座標系2: (μ, σ²)
    ax2 = axes[1]
    for mu in mu_vals:
        for sigma in sigma_vals:
            ax2.plot(mu, sigma**2, 'g.', markersize=4)
    ax2.set_xlabel('μ')
    ax2.set_ylabel('σ² (分散)')
    ax2.set_title('座標系2: (μ, σ²)')
    ax2.grid(True, alpha=0.3)

    # 座標系3: (μ, 1/σ²) - 精度パラメータ
    ax3 = axes[2]
    for mu in mu_vals:
        for sigma in sigma_vals:
            ax3.plot(mu, 1/sigma**2, 'r.', markersize=4)
    ax3.set_xlabel('μ')
    ax3.set_ylabel('τ = 1/σ² (精度)')
    ax3.set_title('座標系3: (μ, τ)\nベイズ推定で頻出')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('manifold_basics_02_coordinates.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. 同じ多様体を異なる座標系で記述できる
2. ベイズ推定では精度 τ = 1/σ² を使うことが多い
   - 事前分布と尤度の積が計算しやすい
3. 情報幾何では座標系に依存しない「不変な」量を扱う
   - Fisher情報行列 → リーマン計量
   - KLダイバージェンス → ダイバージェンス関数
""")


# =============================================================================
# 3. なぜ「多様体」という言葉を使うのか
# =============================================================================

def plot_why_manifold():
    """
    なぜパラメータ空間を「多様体」と呼ぶのか

    球面の例と比較：
    - 球面は3D空間に埋め込まれた2次元多様体
    - 正規分布のパラメータ空間も、実は曲がった空間
    - Fisher計量を入れると、この「曲がり」が見える
    """
    fig = plt.figure(figsize=(14, 5))

    # 左図：球面（2次元多様体の典型例）
    ax1 = fig.add_subplot(121, projection='3d')
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.6, cmap='viridis')
    ax1.set_title('球面：3D空間内の2D多様体\n局所的には平面に見える', fontsize=11)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # 右図：正規分布の多様体を「曲がった空間」として描画
    # Fisher計量による歪みを示唆する図
    ax2 = fig.add_subplot(122)

    # ユークリッド的なグリッド
    mu_range = np.linspace(-2, 2, 11)
    sigma_range = np.linspace(0.3, 2, 11)

    # 通常のグリッド（薄い線）
    for mu in mu_range:
        ax2.plot([mu, mu], [0.3, 2], 'b-', alpha=0.2)
    for sigma in sigma_range:
        ax2.plot([-2, 2], [sigma, sigma], 'b-', alpha=0.2)

    # Fisher計量による「等距離線」（概念的）
    # σが小さいほど、μ方向の距離が大きくなることを示唆
    for i, sigma in enumerate([0.5, 1.0, 1.5]):
        # σが小さいほど密なグリッド
        mu_dense = np.linspace(-2, 2, int(20/sigma))
        for mu in mu_dense:
            ax2.plot(mu, sigma, 'r.', markersize=3)

    ax2.annotate('σが小さい領域：\n「Fisher的に」密',
                 xy=(0, 0.5), xytext=(1.5, 1.2),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    ax2.set_xlabel('μ', fontsize=12)
    ax2.set_ylabel('σ', fontsize=12)
    ax2.set_title('正規分布の多様体\nFisher計量では「曲がって」いる', fontsize=11)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(0, 2.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('manifold_basics_03_why_manifold.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
【学習ポイント】
1. 多様体 = 局所的にユークリッド空間のように見える空間
2. 球面は3D空間内の2D多様体（曲がっている）
3. 統計的多様体も「Fisher計量」を入れると曲がった空間になる
4. σが小さい領域は「Fisher的に」遠い
   - 小さいσの分布を少し変えると大きな情報変化
   - カルマンフィルタで観測精度が高いほど更新が大きいのと同じ直感
""")


if __name__ == '__main__':
    print("=" * 60)
    print("§1.1 微分可能多様体 - 正規分布を例に理解する")
    print("=" * 60)

    print("\n[1/3] パラメータ空間を多様体として可視化")
    plot_parameter_space_as_manifold()

    print("\n[2/3] 局所座標系の概念")
    plot_local_coordinates()

    print("\n[3/3] なぜ「多様体」という言葉を使うのか")
    plot_why_manifold()

    print("\n" + "=" * 60)
    print("次のステップ: sec02_tangent_vectors.py で接ベクトルを学ぶ")
    print("=" * 60)
