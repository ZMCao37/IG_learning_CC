# 情報幾何学習プロジェクト (Information Geometry Learning)

甘利俊一「情報幾何学の基礎」を読み進めながら、各概念を可視化・実装で理解するプロジェクト。

## 学習者の背景

- **数学**: 線形代数・微積分は習得済み、微分幾何は初学者
- **研究経験**: ベイズ推定、KLダイバージェンス、カルマンフィルタ(EKF)、期待効用最大化
- **目標**: 機械学習の最新研究を読み解く基礎を構築

## プロジェクト構造

```
├── chapters/
│   ├── ch00_prerequisites/          # 第0章：前置知識の復習
│   │   ├── 00_overview.ipynb           # 全体概要
│   │   ├── 01_linear_algebra.ipynb     # 線形代数の復習
│   │   ├── 02_probability_statistics.ipynb  # 確率・統計の復習
│   │   └── 03_kl_divergence_fisher.ipynb    # KLダイバージェンスとFisher情報
│   ├── ch01_differential_geometry/  # 第1章：微分幾何の基礎
│   │   ├── sec01_manifold_basics.ipynb     # §1.1 多様体の基礎
│   │   ├── sec02_tangent_vectors.ipynb     # §1.2 接ベクトルと接空間
│   │   └── sec03_metric_tensor.ipynb       # §1.5 リーマン計量
│   ├── ch02_statistical_models/     # 第2章：統計的モデルの幾何（準備中）
│   ├── ch03_dual_connections/       # 第3章：双対接続の理論（準備中）
│   └── ch04_statistical_inference/  # 第4章：統計的推論の微分幾何（準備中）
├── interactive/                 # インタラクティブHTML
├── notes/                       # 学習ノート（Markdown）
└── prompts/                     # 有効だったプロンプト記録
```

## 学習の流れ

### Step 0: 前置知識の復習（ch00）

既知の概念と情報幾何の対応を確認：

| ノートブック | 内容 | 情報幾何との接続 |
|------------|------|----------------|
| `01_linear_algebra.ipynb` | 線形代数の復習 | テンソル、計量、双対空間 |
| `02_probability_statistics.ipynb` | 確率・統計の復習 | 統計的多様体の定義 |
| `03_kl_divergence_fisher.ipynb` | KLダイバージェンスとFisher情報 | **リーマン計量、ダイバージェンス** |

### Step 1: 微分幾何の基礎（ch01）

| 情報幾何の概念 | 既知の概念との対応 | ノートブック |
|------------|----------------|------------|
| 多様体 | パラメータ空間（μ, σの空間） | `sec01_manifold_basics.ipynb` |
| 接ベクトル | パラメータの微小変化の方向 | `sec02_tangent_vectors.ipynb` |
| リーマン計量 | Fisher情報行列 | `sec03_metric_tensor.ipynb` |

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

```bash
# Jupyter Notebookを起動
jupyter notebook

# chapters/ch00_prerequisites/00_overview.ipynb から開始
```

各ノートブックは：
- 📝 概念の説明
- 💻 Pythonコードでの実装・可視化
- 🔗 情報幾何との接続ポイント
- ❓ 確認問題

の形式で構成されています。

## 参考文献

- 甘利俊一「情報幾何学の基礎」（主教材）
- 甘利俊一「情報幾何学の新展開」
- Amari, S. "Information Geometry and Its Applications" (Springer, 2016)
