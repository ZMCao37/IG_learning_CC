# 情報幾何学習プロジェクト (Information Geometry Learning)

甘利俊一「情報幾何学の方法」を読み進めながら、各概念を可視化・実装で理解するプロジェクト。

## 学習者の背景

- **数学**: 線形代数・微積分は習得済み、微分幾何は初学者
- **研究経験**: ベイズ推定、KLダイバージェンス、カルマンフィルタ(EKF)、期待効用最大化
- **目標**: 機械学習の最新研究を読み解く基礎を構築

## プロジェクト構造

```
info-geometry-learning/
│
├── README.md                     # プロジェクト概要、学習ロードマップ
├── requirements.txt              # 依存パッケージ
│
├── notebooks/                    # メインの学習ノート
│   ├── ch00_prerequisites/       # 第0章：前置知識の復習
│   │   ├── 00_overview.ipynb        # 全体概要
│   │   ├── 01_linear_algebra.ipynb  # 線形代数の復習
│   │   ├── 02_probability_statistics.ipynb  # 確率・統計の復習
│   │   └── 03_kl_divergence_fisher.ipynb    # KLダイバージェンスとFisher情報
│   │
│   ├── ch01_differential_geometry/  # 第1章：微分幾何の基礎
│   │   ├── 01_manifolds.ipynb          # §1.1 多様体の基礎
│   │   ├── 02_tangent_vectors.ipynb    # §1.2 接ベクトルと接空間
│   │   ├── 05_riemannian_metric.ipynb  # §1.5 リーマン計量
│   │   └── 06_affine_connection.ipynb  # §1.6 アフィン接続（準備中）
│   │
│   ├── ch02_statistical_models/     # 第2章：統計的モデルの幾何（準備中）
│   │   ├── 01_statistical_model.ipynb   # 統計的モデル
│   │   ├── 02_fisher_information.ipynb  # Fisher情報
│   │   └── 03_alpha_connection.ipynb    # α接続
│   │
│   └── ch03_dual_connections/       # 第3章：双対接続の理論（準備中）
│       ├── 01_duality.ipynb            # 双対性
│       ├── 02_dual_flat_space.ipynb    # 双対平坦空間
│       └── 03_divergence.ipynb         # ダイバージェンス
│
├── src/                          # 再利用可能なコード
│   ├── __init__.py
│   ├── distributions.py          # 確率分布クラス
│   ├── manifolds.py              # 統計多様体クラス
│   └── visualization.py          # 可視化ユーティリティ
│
├── interactive/                  # インタラクティブHTML版
│   └── fisher_ellipse.html
│
└── prompts/                      # 有効だったプロンプト記録
    ├── concept_introduction.md      # 概念導入のパターン
    ├── visualization_patterns.md    # 可視化のパターン
    └── debugging_patterns.md        # デバッグのパターン
```

## 学習の流れ

### Step 0: 前置知識の復習（ch00）

既知の概念と情報幾何の対応を確認：

| ノートブック | 内容 | 情報幾何との接続 |
|------------|------|-----------------|
| `01_linear_algebra.ipynb` | 線形代数の復習 | テンソル、計量、双対空間 |
| `02_probability_statistics.ipynb` | 確率・統計の復習 | 統計的多様体の定義 |
| `03_kl_divergence_fisher.ipynb` | KLダイバージェンスとFisher情報 | **リーマン計量、ダイバージェンス** |

### Step 1: 微分幾何の基礎（ch01）

| 情報幾何の概念 | 既知の概念との対応 | ノートブック |
|------------|----------------|-------------|
| 多様体 | パラメータ空間（μ, σの空間） | `01_manifolds.ipynb` |
| 接ベクトル | パラメータの微小変化の方向 | `02_tangent_vectors.ipynb` |
| リーマン計量 | Fisher情報行列 | `05_riemannian_metric.ipynb` |

### Step 2以降: 教科書本論（準備中）

- 第2章：統計的モデルの幾何学的構造
- 第3章：双対接続とダイバージェンス
- 第4章以降：応用

## 環境セットアップ

```bash
pip install -r requirements.txt
# または
pip install numpy matplotlib scipy plotly jupyter
```

## 使い方

### ノートブックで学習

```bash
# Jupyter Notebookを起動
jupyter notebook

# notebooks/ch00_prerequisites/00_overview.ipynb から開始
```

### srcモジュールの利用

```python
# プロジェクトルートで
from src import GaussianDistribution, GaussianManifold, plot_fisher_ellipse

# 正規分布の作成
dist = GaussianDistribution(mu=0, sigma=1)
print(dist.fisher_information())

# 多様体上の計算
manifold = GaussianManifold()
theta = [0, 1]  # (μ, σ)
I = manifold.fisher_metric(theta)
```

## ノートブックの構成

各ノートブックは統一されたテンプレートで構成：

1. **概要** - この節で学ぶこと、既知概念との対応
2. **直感的理解** - 身近な例からの理解
3. **数学的定義** - 厳密な定義と公式
4. **可視化** - Python/matplotlibによる図示
5. **具体例** - 計算例とコード
6. **他の概念との関係** - カルマンフィルタ、ベイズ推定等との接続
7. **演習問題** - 折りたたみ解答付き
8. **参考：使用したプロンプト** - 有効だったプロンプト例

## 参考文献

- 甘利俊一・長岡浩司「情報幾何学の方法」、岩波書店（主教材）
- 甘利俊一「情報幾何学の新展開」
- Amari, S. "Information Geometry and Its Applications" (Springer, 2016)

## コントリビューション

質問やフィードバックは Issues にてお願いします。
