#!/usr/bin/env python3
"""
AIグラウンディング手法マップ — PDF解説資料生成スクリプト

IPAゴシックフォントを使用して日本語PDF資料を生成する。
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ==============================================================================
# フォント設定
# ==============================================================================
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
FONT_PATH_P = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"

pdfmetrics.registerFont(TTFont('IPAGothic', FONT_PATH))
pdfmetrics.registerFont(TTFont('IPAPGothic', FONT_PATH_P))

# matplotlib用フォント設定
fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# カラーパレット
# ==============================================================================
C_PRIMARY    = HexColor('#1a365d')   # ダークネイビー
C_SECONDARY  = HexColor('#2b6cb0')   # ブルー
C_ACCENT     = HexColor('#e53e3e')   # レッド
C_BG_LIGHT   = HexColor('#f7fafc')   # 薄グレー
C_BG_HEADER  = HexColor('#2d3748')   # ダークグレー
C_TEXT       = HexColor('#1a202c')   # ほぼ黒
C_TEXT_LIGHT = HexColor('#4a5568')   # グレーテキスト
C_BORDER     = HexColor('#e2e8f0')   # ボーダー

# 象限カラー
C_Q1 = HexColor('#2b6cb0')  # 右上 - 青
C_Q2 = HexColor('#2f855a')  # 左上 - 緑
C_Q3 = HexColor('#d69e2e')  # 左下 - 黄
C_Q4 = HexColor('#c53030')  # 右下 - 赤

# カテゴリカラー
CAT_COLORS = {
    'A': '#2f855a',  # 緑: 形式的知識表現
    'B': '#2b6cb0',  # 青: 検索・文書ベース
    'C': '#d69e2e',  # 黄: リアルタイム接続
    'D': '#9b2c2c',  # 赤: 暗黙的・学習
    'E': '#6b46c1',  # 紫: ハイブリッド
}

# ==============================================================================
# スタイル定義
# ==============================================================================
def make_styles():
    styles = {}
    styles['title'] = ParagraphStyle(
        'Title', fontName='IPAGothic', fontSize=28, leading=36,
        textColor=white, alignment=TA_CENTER, spaceAfter=6*mm
    )
    styles['subtitle'] = ParagraphStyle(
        'Subtitle', fontName='IPAPGothic', fontSize=14, leading=20,
        textColor=HexColor('#a0aec0'), alignment=TA_CENTER, spaceAfter=4*mm
    )
    styles['h1'] = ParagraphStyle(
        'H1', fontName='IPAGothic', fontSize=20, leading=28,
        textColor=C_PRIMARY, spaceBefore=8*mm, spaceAfter=4*mm,
        borderPadding=(0, 0, 2*mm, 0)
    )
    styles['h2'] = ParagraphStyle(
        'H2', fontName='IPAGothic', fontSize=15, leading=22,
        textColor=C_SECONDARY, spaceBefore=6*mm, spaceAfter=3*mm
    )
    styles['h3'] = ParagraphStyle(
        'H3', fontName='IPAGothic', fontSize=12, leading=18,
        textColor=C_TEXT, spaceBefore=4*mm, spaceAfter=2*mm
    )
    styles['body'] = ParagraphStyle(
        'Body', fontName='IPAPGothic', fontSize=9.5, leading=16,
        textColor=C_TEXT, spaceAfter=2*mm, alignment=TA_JUSTIFY
    )
    styles['body_small'] = ParagraphStyle(
        'BodySmall', fontName='IPAPGothic', fontSize=8.5, leading=14,
        textColor=C_TEXT_LIGHT, spaceAfter=1.5*mm
    )
    styles['bullet'] = ParagraphStyle(
        'Bullet', fontName='IPAPGothic', fontSize=9.5, leading=15,
        textColor=C_TEXT, leftIndent=8*mm, bulletIndent=3*mm,
        spaceAfter=1*mm
    )
    styles['caption'] = ParagraphStyle(
        'Caption', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT_LIGHT, alignment=TA_CENTER, spaceAfter=2*mm
    )
    styles['toc'] = ParagraphStyle(
        'TOC', fontName='IPAPGothic', fontSize=11, leading=20,
        textColor=C_PRIMARY, leftIndent=5*mm
    )
    styles['page_num'] = ParagraphStyle(
        'PageNum', fontName='IPAPGothic', fontSize=8, textColor=C_TEXT_LIGHT,
        alignment=TA_CENTER
    )
    styles['table_header'] = ParagraphStyle(
        'TableHeader', fontName='IPAGothic', fontSize=8.5, leading=12,
        textColor=white, alignment=TA_CENTER
    )
    styles['table_cell'] = ParagraphStyle(
        'TableCell', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT
    )
    styles['table_cell_center'] = ParagraphStyle(
        'TableCellCenter', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT, alignment=TA_CENTER
    )
    styles['quadrant_title'] = ParagraphStyle(
        'QuadrantTitle', fontName='IPAGothic', fontSize=11, leading=16,
        textColor=C_PRIMARY, spaceBefore=3*mm, spaceAfter=1.5*mm
    )
    return styles


# ==============================================================================
# ページテンプレート
# ==============================================================================
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('IPAPGothic', 8)
    canvas.setFillColor(C_TEXT_LIGHT)
    page_num = canvas.getPageNumber()
    if page_num > 1:
        canvas.drawCentredString(A4[0]/2, 12*mm, f"— {page_num} —")
        # ヘッダーライン
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(20*mm, A4[1] - 15*mm, A4[0] - 20*mm, A4[1] - 15*mm)
        # フッターライン
        canvas.line(20*mm, 18*mm, A4[0] - 20*mm, 18*mm)
    canvas.restoreState()


# ==============================================================================
# 地図1: 意味の明示度×運用結合度 スキャッタープロット生成
# ==============================================================================
def generate_map1_chart(output_path):
    techniques = [
        ("形式論理/\nルールエンジン",   2.0, 5.0, 'A'),
        ("OWLオントロジー",            2.5, 4.7, 'A'),
        ("デジタルツイン",              5.0, 4.8, 'C'),
        ("ナレッジグラフ",              2.5, 4.2, 'A'),
        ("Graph RAG",                  3.5, 4.3, 'E'),
        ("ツール利用/MCP",             4.5, 4.0, 'C'),
        ("セマンティック\nレイヤー",     3.0, 4.0, 'E'),
        ("タクソノミー/\n統制語彙",      1.5, 3.2, 'A'),
        ("スキーマ/\nデータモデル",      2.5, 3.0, 'A'),
        ("Advanced RAG",              3.5, 3.0, 'B'),
        ("ガードレール/\nConstitutional", 4.5, 3.2, 'D'),
        ("Feature Store",             4.5, 3.0, 'E'),
        ("構造化\nプロンプティング",     1.0, 2.5, 'B'),
        ("エンベディング\n検索",         2.5, 2.2, 'B'),
        ("Naive RAG",                  3.0, 2.0, 'B'),
        ("マルチモーダル",              4.5, 2.2, 'C'),
        ("メモリシステム",              2.8, 2.0, 'E'),
        ("平ドキュメント",              1.0, 1.2, 'B'),
        ("ファインチューニング\n/RLHF",  2.0, 1.0, 'D'),
    ]

    fig, ax = plt.subplots(figsize=(11, 8))

    # 象限の背景
    ax.axhline(y=3.0, color='#cbd5e0', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axvline(x=3.0, color='#cbd5e0', linewidth=0.8, linestyle='--', alpha=0.7)

    # 象限ラベル
    ax.text(1.5, 4.7, '象限II: 知識の宝庫', fontsize=10, ha='center',
            color='#2f855a', alpha=0.5, fontweight='bold')
    ax.text(4.2, 4.7, '象限I: デジタル統治', fontsize=10, ha='center',
            color='#2b6cb0', alpha=0.5, fontweight='bold')
    ax.text(1.5, 1.3, '象限III: 軽量スタート', fontsize=10, ha='center',
            color='#d69e2e', alpha=0.5, fontweight='bold')
    ax.text(4.2, 1.3, '象限IV: ブラックボックス接続', fontsize=10, ha='center',
            color='#c53030', alpha=0.5, fontweight='bold')

    # 背景塗り
    ax.fill_between([0.5, 3.0], 3.0, 5.5, alpha=0.04, color='#2f855a')
    ax.fill_between([3.0, 5.5], 3.0, 5.5, alpha=0.04, color='#2b6cb0')
    ax.fill_between([0.5, 3.0], 0.5, 3.0, alpha=0.04, color='#d69e2e')
    ax.fill_between([3.0, 5.5], 0.5, 3.0, alpha=0.04, color='#c53030')

    cat_labels = {
        'A': '形式的知識表現',
        'B': '検索・文書ベース',
        'C': 'リアルタイム接続',
        'D': '暗黙的・学習系',
        'E': 'ハイブリッド・新興',
    }

    plotted_cats = set()
    for name, x, y, cat in techniques:
        color = CAT_COLORS[cat]
        label = cat_labels[cat] if cat not in plotted_cats else None
        plotted_cats.add(cat)
        ax.scatter(x, y, c=color, s=120, zorder=5, edgecolors='white',
                   linewidth=1.2, label=label, alpha=0.9)
        # テキスト位置の微調整
        offset_x, offset_y = 0.08, 0.12
        if 'Graph RAG' in name:
            offset_x, offset_y = 0.12, 0.15
        elif 'Naive' in name:
            offset_x, offset_y = 0.12, -0.18
        elif 'Advanced' in name:
            offset_y = 0.15
        elif 'メモリ' in name:
            offset_x, offset_y = 0.12, -0.15
        elif 'Feature' in name:
            offset_x, offset_y = 0.12, -0.15
        elif 'エンベディング' in name:
            offset_y = -0.2
        ax.annotate(name, (x, y), fontsize=7, color='#2d3748',
                    xytext=(x + offset_x, y + offset_y), fontweight='bold')

    # 進化の矢印
    arrows = [
        (3.0, 2.0, 3.5, 3.0, '#718096'),   # Naive RAG → Advanced RAG
        (3.5, 3.0, 3.5, 4.3, '#718096'),   # Advanced RAG → Graph RAG
        (1.0, 1.2, 3.0, 2.0, '#a0aec0'),   # 平ドキュメント → Naive RAG
    ]
    for x1, y1, x2, y2, color in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.2, alpha=0.5,
                                   connectionstyle='arc3,rad=0.15'))

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel('運用結合度・ガバナンス性  →', fontsize=11, labelpad=10)
    ax.set_ylabel('意味の明示度・形式性  →', fontsize=11, labelpad=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['低', '', '中', '', '高'])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['低', '', '中', '', '高'])
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
              edgecolor='#e2e8f0')
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ==============================================================================
# 地図2: 知識の抽象度×応答レイテンシ スキャッタープロット生成
# ==============================================================================
def generate_map2_chart(output_path):
    techniques = [
        ("OWLオントロジー",           1.0, 5.0, 'A'),
        ("形式論理/\nルールエンジン",   1.2, 4.8, 'A'),
        ("ガードレール/\nConstitutional", 4.5, 5.0, 'D'),
        ("タクソノミー/\n統制語彙",     1.5, 4.2, 'A'),
        ("スキーマ/\nデータモデル",     2.5, 4.0, 'A'),
        ("セマンティック\nレイヤー",    4.0, 4.0, 'E'),
        ("ファインチューニング\n/RLHF", 2.0, 3.5, 'D'),
        ("ナレッジグラフ",             2.5, 3.2, 'A'),
        ("Graph RAG",                 3.5, 3.2, 'E'),
        ("構造化\nプロンプティング",    1.5, 3.0, 'B'),
        ("Advanced RAG",              3.5, 2.8, 'B'),
        ("エンベディング\n検索",        2.5, 2.2, 'B'),
        ("Naive RAG",                 3.5, 2.0, 'B'),
        ("ツール利用/MCP",            5.0, 2.0, 'C'),
        ("平ドキュメント",             1.0, 2.0, 'B'),
        ("メモリシステム",             2.5, 1.8, 'E'),
        ("Feature Store",            4.0, 1.5, 'E'),
        ("デジタルツイン",             5.0, 1.0, 'C'),
        ("マルチモーダル",             4.0, 1.0, 'C'),
    ]

    fig, ax = plt.subplots(figsize=(11, 8))

    # ゾーンの背景
    # 設計時知識ゾーン
    from matplotlib.patches import FancyBboxPatch
    rect1 = FancyBboxPatch((0.5, 3.5), 4.8, 2.0, boxstyle="round,pad=0.1",
                           facecolor='#ebf8ff', edgecolor='#90cdf4',
                           linewidth=1, alpha=0.5)
    ax.add_patch(rect1)
    ax.text(2.9, 5.3, '設計時知識ゾーン', fontsize=10, ha='center',
            color='#2b6cb0', fontweight='bold', alpha=0.7)
    ax.text(2.9, 5.05, '（月〜年単位で更新）', fontsize=7, ha='center',
            color='#4299e1', alpha=0.6)

    # 蓄積知識ゾーン
    rect2 = FancyBboxPatch((0.5, 1.3), 2.3, 2.1, boxstyle="round,pad=0.1",
                           facecolor='#f0fff4', edgecolor='#9ae6b4',
                           linewidth=1, alpha=0.5)
    ax.add_patch(rect2)
    ax.text(1.65, 3.15, '蓄積知識ゾーン', fontsize=9, ha='center',
            color='#2f855a', fontweight='bold', alpha=0.7)

    # 動的知識ゾーン
    rect3 = FancyBboxPatch((3.0, 0.5), 2.3, 2.9, boxstyle="round,pad=0.1",
                           facecolor='#fffff0', edgecolor='#fefcbf',
                           linewidth=1, alpha=0.5)
    ax.add_patch(rect3)
    ax.text(4.15, 3.15, '動的知識ゾーン', fontsize=9, ha='center',
            color='#b7791f', fontweight='bold', alpha=0.7)
    ax.text(4.15, 2.9, '（秒〜分で更新）', fontsize=7, ha='center',
            color='#d69e2e', alpha=0.6)

    cat_labels = {
        'A': '形式的知識表現',
        'B': '検索・文書ベース',
        'C': 'リアルタイム接続',
        'D': '暗黙的・学習系',
        'E': 'ハイブリッド・新興',
    }

    plotted_cats = set()
    for name, x, y, cat in techniques:
        color = CAT_COLORS[cat]
        label = cat_labels[cat] if cat not in plotted_cats else None
        plotted_cats.add(cat)
        ax.scatter(x, y, c=color, s=120, zorder=5, edgecolors='white',
                   linewidth=1.2, label=label, alpha=0.9)
        offset_x, offset_y = 0.1, 0.13
        if 'ガードレール' in name:
            offset_x, offset_y = -0.6, 0.15
        elif 'Naive' in name:
            offset_y = -0.18
        elif 'メモリ' in name:
            offset_y = -0.15
        ax.annotate(name, (x, y), fontsize=7, color='#2d3748',
                    xytext=(x + offset_x, y + offset_y), fontweight='bold')

    ax.set_xlim(0.3, 5.8)
    ax.set_ylim(0.3, 5.7)
    ax.set_xlabel('応答レイテンシ / 知識の鮮度  →（リアルタイム）', fontsize=11, labelpad=10)
    ax.set_ylabel('知識の抽象度  →（概念・スキーマ）', fontsize=11, labelpad=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['事前準備型\n（静的）', 'バッチ\n更新', '中間', '準リアル\nタイム', 'リアル\nタイム'])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['具体的\n(データ)', '', '中間', '', '抽象的\n(スキーマ)'])
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9, edgecolor='#e2e8f0')
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ==============================================================================
# 地図3: パイプラインカバレッジヒートマップ生成
# ==============================================================================
def generate_map3_chart(output_path):
    stages = ['ソース', '構造化', '格納\n索引化', '検索\n取得', 'コンテキスト\n注入', '生成\n推論', '出力\n制御', 'フィード\nバック']
    techniques_data = [
        ('OWLオントロジー',         [0, 1, 1, 0, 0, 0, 1, 0], 'A'),
        ('ナレッジグラフ',          [0, 1, 1, 1, 0, 0, 0, 1], 'A'),
        ('タクソノミー/統制語彙',    [0, 1, 0, 0, 0, 0, 0, 0], 'A'),
        ('スキーマ/データモデル',    [0, 1, 0, 0, 0, 0, 1, 0], 'A'),
        ('形式論理/ルールエンジン',  [0, 1, 0, 0, 0, 0, 1, 0], 'A'),
        ('RAG各種',               [0, 0, 1, 1, 1, 0, 0, 0], 'B'),
        ('平ドキュメント',          [1, 0, 0, 0, 1, 0, 0, 0], 'B'),
        ('エンベディング検索',      [0, 0, 1, 1, 0, 0, 0, 0], 'B'),
        ('構造化プロンプティング',   [0, 0, 0, 0, 1, 0, 0, 0], 'B'),
        ('ツール利用/MCP',         [1, 0, 0, 0, 0, 1, 0, 0], 'C'),
        ('デジタルツイン',          [1, 0, 0, 0, 0, 1, 0, 0], 'C'),
        ('マルチモーダル',          [1, 0, 0, 0, 0, 1, 0, 0], 'C'),
        ('ファインチューニング/RLHF', [0, 0, 0, 0, 0, 1, 0, 1], 'D'),
        ('ガードレール/Constitutional', [0, 0, 0, 0, 0, 0, 1, 0], 'D'),
        ('セマンティックレイヤー',   [0, 1, 1, 1, 0, 0, 0, 0], 'E'),
        ('Feature Store',          [0, 0, 1, 0, 0, 0, 0, 1], 'E'),
        ('メモリシステム',          [0, 0, 1, 0, 1, 0, 0, 1], 'E'),
    ]

    names = [t[0] for t in techniques_data]
    data = np.array([t[1] for t in techniques_data])
    cats = [t[2] for t in techniques_data]

    fig, ax = plt.subplots(figsize=(12, 8))

    # カスタムカラーマップ
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#f7fafc', '#3182ce'])

    ax.imshow(data, cmap=cmap, aspect='auto', alpha=0.8)

    # セルのテキストとボーダー
    for i in range(len(names)):
        for j in range(len(stages)):
            if data[i, j] == 1:
                ax.text(j, i, '●', ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold')
            # セルボーダー
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor='#e2e8f0', linewidth=0.5)
            ax.add_patch(rect)

    # カテゴリ色の帯（左端）
    for i, cat in enumerate(cats):
        rect = plt.Rectangle((-0.5, i-0.5), 0.15, 1, fill=True,
                             facecolor=CAT_COLORS[cat], alpha=0.8)
        ax.add_patch(rect)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_xlabel('')

    # ステージ番号
    for j, stage in enumerate(stages):
        ax.text(j, -1.2, f'①②③④⑤⑥⑦⑧'[j], ha='center', va='center',
                fontsize=10, color='#4a5568', fontweight='bold')

    # パイプラインフロー矢印
    for j in range(len(stages) - 1):
        ax.annotate('', xy=(j+0.7, -1.2), xytext=(j+0.3, -1.2),
                    arrowprops=dict(arrowstyle='->', color='#a0aec0', lw=1.5))

    ax.set_xlim(-0.5, len(stages)-0.5)
    ax.set_ylim(len(names)-0.5, -1.5)

    # 凡例
    cat_labels = {
        'A': '形式的知識表現', 'B': '検索・文書ベース',
        'C': 'リアルタイム接続', 'D': '暗黙的・学習系',
        'E': 'ハイブリッド・新興'
    }
    legend_elements = []
    for cat, label in cat_labels.items():
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=CAT_COLORS[cat], label=label, alpha=0.8))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='#e2e8f0')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ==============================================================================
# PDF文書構築
# ==============================================================================
def build_pdf(output_path):
    s = make_styles()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=22*mm,
        title='AIグラウンディング手法マップ 解説資料',
        author='Grounding Research Team'
    )

    story = []
    W = A4[0] - 40*mm  # 有効幅

    # ==== 表紙 ====
    story.append(Spacer(1, 50*mm))

    # 表紙背景テーブル
    title_data = [[
        Paragraph('AIグラウンディング手法マップ', s['title']),
    ], [
        Paragraph('— 解説資料 —', s['subtitle']),
    ], [
        Spacer(1, 8*mm),
    ], [
        Paragraph('現実世界の知識をAIに橋渡しする18手法の網羅的調査と可視化', s['subtitle']),
    ]]
    title_table = Table(title_data, colWidths=[W])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_BG_HEADER),
        ('TOPPADDING', (0, 0), (-1, -1), 8*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 5*mm),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5*mm),
        ('ROUNDEDCORNERS', [3*mm, 3*mm, 3*mm, 3*mm]),
    ]))
    story.append(title_table)

    story.append(Spacer(1, 15*mm))

    # メタ情報
    meta_style = ParagraphStyle(
        'Meta', fontName='IPAPGothic', fontSize=10, leading=16,
        textColor=C_TEXT_LIGHT, alignment=TA_CENTER
    )
    story.append(Paragraph('2026年3月', meta_style))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph('対象手法: 18手法（5群分類）', meta_style))
    story.append(Paragraph('地図パターン: 3種', meta_style))

    story.append(PageBreak())

    # ==== 目次 ====
    story.append(Paragraph('目次', s['h1']))
    story.append(Spacer(1, 3*mm))

    toc_items = [
        '1. グラウンディングとは何か',
        '2. 手法の分類体系（5群18手法）',
        '3. 地図1: 意味の明示度 × 運用結合度・ガバナンス性',
        '4. 地図2: 知識の抽象度 × 応答レイテンシ',
        '5. 地図3: データライフサイクル・パイプライン',
        '6. 技術横断比較マトリクス',
        '7. 手法選定ガイドと進化の方向性',
    ]
    for item in toc_items:
        story.append(Paragraph(item, s['toc']))
    story.append(PageBreak())

    # ==== 1. グラウンディングとは何か ====
    story.append(Paragraph('1. グラウンディングとは何か', s['h1']))
    story.append(Paragraph(
        'AIグラウンディングとは、大規模言語モデル（LLM）の応答を現実世界の事実・知識・構造に'
        '根拠付ける（ground）ための技術群の総称である。LLMは訓練データから学んだパターンに基づいて'
        'テキストを生成するが、その知識には時間的なカットオフがあり、また事実と異なる情報を自信を持って'
        '生成する「ハルシネーション」が発生し得る。', s['body']))
    story.append(Paragraph(
        'グラウンディング手法は、この問題に対処するために現実世界の知識をAIに橋渡しする役割を担う。'
        '平ドキュメントのような最もシンプルなものから、オントロジーやデジタルツインのような高度に形式化'
        'されたものまで、多様な手法が存在し、それぞれ異なるトレードオフを持つ。', s['body']))
    story.append(Paragraph(
        '本資料では、これらの手法を網羅的に調査・分類し、3種類の「地図」を用いて手法間の関係性と'
        '特性の違いを可視化する。', s['body']))

    # なぜ地図が必要か
    story.append(Paragraph('なぜ地図が必要か', s['h2']))
    why_data = [
        ['課題', '地図が提供する解決'],
        ['手法が多すぎて選べない', '2軸のマッピングで手法の位置づけを直感的に把握'],
        ['技術原理の違いがわかりにくい', '形式性・結合度等の軸で技術的特性を比較可能'],
        ['組み合わせ方がわからない', 'パイプライン図で各手法の守備範囲と相互補完性を可視化'],
        ['進化の方向性が見えない', '地図上の進化矢印でトレンドと成熟度の進路を表現'],
    ]
    why_table = Table(why_data, colWidths=[W*0.35, W*0.65])
    why_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_HEADER),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'IPAGothic'),
        ('FONTNAME', (0, 1), (-1, -1), 'IPAPGothic'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEADING', (0, 0), (-1, -1), 14),
        ('TOPPADDING', (0, 0), (-1, -1), 3*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 3*mm),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, C_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(why_table)

    story.append(PageBreak())

    # ==== 2. 手法の分類体系 ====
    story.append(Paragraph('2. 手法の分類体系（5群18手法）', s['h1']))
    story.append(Paragraph(
        'グラウンディング手法を知識の形式化度とシステム結合方式に基づき5群に分類する。', s['body']))

    # 分類表
    cat_header = [
        Paragraph('群', s['table_header']),
        Paragraph('分類名', s['table_header']),
        Paragraph('含まれる手法', s['table_header']),
        Paragraph('特徴', s['table_header']),
    ]
    cat_data = [cat_header]
    categories = [
        ('A', '形式的\n知識表現',
         'OWLオントロジー、ナレッジグラフ、\nタクソノミー/統制語彙、スキーマ/\nデータモデル、形式論理/ルールエンジン',
         '意味を機械可読な形式\nで厳密に定義'),
        ('B', '検索・\n文書ベース',
         'RAG各種(Naive/Advanced/\nModular/Graph/Agentic)、\n平ドキュメント、エンベディング\n検索、構造化プロンプティング',
         'テキスト・文書から動的\nに知識を検索・注入'),
        ('C', 'リアルタイム\n接続',
         'ツール利用/MCP、\nデジタルツイン、\nマルチモーダルグラウンディング',
         '外部システム・物理世界\nとのリアルタイム接続'),
        ('D', '暗黙的・\n学習系',
         'ファインチューニング/RLHF、\nガードレール/Constitutional AI',
         'モデルパラメータや\n行動ルールに知識を内包'),
        ('E', 'ハイブリッド\n・新興',
         'Graph RAG、セマンティック\nレイヤー、Feature Store、\nメモリシステム',
         '複数手法の合成による\n新しいアプローチ'),
    ]
    for group, name, techs, feature in categories:
        cat_data.append([
            Paragraph(f'<b>{group}</b>', s['table_cell_center']),
            Paragraph(name, s['table_cell']),
            Paragraph(techs, s['table_cell']),
            Paragraph(feature, s['table_cell']),
        ])

    cat_table = Table(cat_data, colWidths=[W*0.06, W*0.14, W*0.45, W*0.35])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_HEADER),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 2*mm),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, C_BG_LIGHT]),
        # カテゴリ色
        ('BACKGROUND', (0, 1), (0, 1), HexColor(CAT_COLORS['A'])),
        ('TEXTCOLOR', (0, 1), (0, 1), white),
        ('BACKGROUND', (0, 2), (0, 2), HexColor(CAT_COLORS['B'])),
        ('TEXTCOLOR', (0, 2), (0, 2), white),
        ('BACKGROUND', (0, 3), (0, 3), HexColor(CAT_COLORS['C'])),
        ('TEXTCOLOR', (0, 3), (0, 3), white),
        ('BACKGROUND', (0, 4), (0, 4), HexColor(CAT_COLORS['D'])),
        ('TEXTCOLOR', (0, 4), (0, 4), white),
        ('BACKGROUND', (0, 5), (0, 5), HexColor(CAT_COLORS['E'])),
        ('TEXTCOLOR', (0, 5), (0, 5), white),
    ]))
    story.append(cat_table)

    # 各手法の1行解説
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph('各手法の概要', s['h2']))

    technique_descs = [
        ('OWLオントロジー', '記述論理に基づきドメインの概念・関係・制約を機械可読形式で厳密に定義。推論器による自動導出が可能'),
        ('ナレッジグラフ', 'エンティティ（ノード）と関係（エッジ）のグラフ構造で知識を表現。マルチホップ推論と説明可能性を提供'),
        ('タクソノミー/統制語彙', '階層的な「is-a」関係で概念を分類。用語の統制と検索精度向上に貢献'),
        ('スキーマ/データモデル', 'JSON Schema等でデータの構造・型・制約を形式的に定義。AIの構造化出力を保証'),
        ('形式論理/ルールエンジン', '述語論理やIF-THENルールで推論規則を宣言的に定義。監査可能な意思決定を実現'),
        ('RAG各種', '外部知識ベースから動的に検索してコンテキストに注入。Naive→Advanced→Graph→Agenticと進化'),
        ('平ドキュメント', 'Markdown/PDF等をそのままコンテキストに注入する最もシンプルな手法。CLAUDE.md等'),
        ('エンベディング検索', 'テキストを密ベクトルに変換し、ベクトル空間上の近傍検索で意味的に類似した内容を取得'),
        ('構造化プロンプティング', 'CoT、Few-shot等のプロンプト設計でLLMの推論を制御。インフラ不要で最も手軽'),
        ('ツール利用/MCP', 'LLMがAPIを介して外部サービスにリアルタイムアクセス。MCPが業界標準として確立'),
        ('デジタルツイン', '物理世界の動的な仮想レプリカ。物理法則に基づくAI予測の検証を可能にする'),
        ('マルチモーダル', '視覚・音声・触覚等のテキスト以外のモダリティでAIの理解を物理世界に固定'),
        ('ファインチューニング/RLHF', 'モデルパラメータに知識を焼き込む暗黙的グラウンディング。スタイル・トーンの制御に有効'),
        ('ガードレール/Constitutional AI', 'AIの行動を安全かつ信頼できる範囲に制約。出力のリアルタイムフィルタリング'),
        ('Graph RAG', 'RAG＋ナレッジグラフの統合。マルチホップ推論とハルシネーション削減を実現'),
        ('セマンティックレイヤー', '技術データをビジネス用語に変換する抽象化層。AIとビジネスの橋渡し'),
        ('Feature Store', 'ML特徴量の保存・管理・提供を一元化。RAG/LLMのグラウンディングレイヤーとしても機能'),
        ('メモリシステム', 'LLMの固定コンテキストウィンドウを超えた持続的記憶。MemGPT/Letta等'),
    ]

    for name, desc in technique_descs:
        story.append(Paragraph(
            f'<b>{name}</b>: {desc}', s['body_small']))

    story.append(PageBreak())

    # ==== 3. 地図1 ====
    story.append(Paragraph('3. 地図1: 意味の明示度 × 運用結合度・ガバナンス性', s['h1']))
    story.append(Paragraph(
        '最も重要な地図。横軸に運用結合度・ガバナンス性（システムとの統合の深さ）、'
        '縦軸に意味の明示度・形式性（知識がどれだけ厳密に形式化されているか）を配置し、'
        '各手法の位置づけを4象限で分析する。', s['body']))

    # チャート生成と挿入
    map1_path = '/tmp/grounding_map1.png'
    generate_map1_chart(map1_path)
    story.append(Image(map1_path, width=W, height=W*0.72))
    story.append(Paragraph('図1: 意味の明示度×運用結合度のスキャッタープロット（色はカテゴリを表す）', s['caption']))

    story.append(PageBreak())

    # 4象限解説
    story.append(Paragraph('4象限の解説', s['h2']))

    quadrants = [
        ('象限I: デジタル統治（右上）— 高形式性 × 高結合度',
         C_Q1,
         '意味が厳密に定義され、かつシステムに深く統合されている。最も制御されたグラウンディング。',
         'デジタルツイン＋物理法則、ツール利用/MCP、Graph RAG、ガードレール/Constitutional AI',
         'ミッションクリティカルなシステム（製造制御、金融取引）、リアルタイム意思決定支援、規制遵守が必須の領域',
         '導入・維持コストが最も高い。柔軟性が低く、変更に伴うリスクが大きい。'),
        ('象限II: 知識の宝庫（左上）— 高形式性 × 低結合度',
         C_Q2,
         '形式的に豊かな知識表現だが、特定のシステムに密結合していない。知識基盤として共有・再利用可能。',
         'OWLオントロジー、形式論理/ルールエンジン、ナレッジグラフ',
         'ドメイン知識のモデリング（医療、法律）、組織横断的な知識共有基盤、推論・整合性検証',
         '構築に専門知識が必要。実運用への橋渡しが別途必要。'),
        ('象限III: 軽量スタート（左下）— 低形式性 × 低結合度',
         C_Q3,
         '手軽に始められ、維持コストが最低。しかし形式性・ガバナンスが弱くスケール限界がある。',
         '平ドキュメント、構造化プロンプティング、ファインチューニング/RLHF',
         'プロトタイプ、PoC、小規模プロジェクト（CLAUDE.md等）、個人利用',
         'スケールしない。品質の保証・監査が困難。'),
        ('象限IV: ブラックボックス接続（右下）— 低形式性 × 高結合度',
         C_Q4,
         'システムに深く統合されているが、意味表現が暗黙的・不透明。',
         'マルチモーダルグラウンディング、Feature Store（一部）',
         'リアルタイムセンサーデータ処理、大規模MLパイプライン',
         '説明可能性が低い。デバッグ・監査が困難。'),
    ]

    for title, color, desc, techs, usecase, tradeoff in quadrants:
        q_data = [
            [Paragraph(f'<b>{title}</b>', ParagraphStyle(
                'QH', fontName='IPAGothic', fontSize=10, leading=15,
                textColor=white))],
            [Paragraph(desc, ParagraphStyle(
                'QD', fontName='IPAPGothic', fontSize=9, leading=14,
                textColor=C_TEXT, spaceBefore=1*mm))],
            [Paragraph(f'<b>主な手法:</b> {techs}', ParagraphStyle(
                'QT', fontName='IPAPGothic', fontSize=8.5, leading=13,
                textColor=C_TEXT))],
            [Paragraph(f'<b>ユースケース:</b> {usecase}', ParagraphStyle(
                'QU', fontName='IPAPGothic', fontSize=8.5, leading=13,
                textColor=C_TEXT))],
            [Paragraph(f'<b>トレードオフ:</b> {tradeoff}', ParagraphStyle(
                'QR', fontName='IPAPGothic', fontSize=8.5, leading=13,
                textColor=C_ACCENT))],
        ]
        q_table = Table(q_data, colWidths=[W])
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), color),
            ('BACKGROUND', (0, 1), (0, -1), C_BG_LIGHT),
            ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2*mm),
            ('LEFTPADDING', (0, 0), (-1, -1), 3*mm),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3*mm),
            ('GRID', (0, 0), (-1, -1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([q_table, Spacer(1, 3*mm)]))

    # 進化の方向性
    story.append(Paragraph('進化の方向性', s['h2']))
    evolutions = [
        'Naive RAG → Advanced RAG → Graph RAG: 形式性と結合度の両方が段階的に上昇',
        '平ドキュメント → RAG → エージェンティックAI: 左下から右上への対角移動',
        'ナレッジグラフ + RAG = Graph RAG: 象限IIから象限Iへの水平移動',
        'マルチモーダル + 形式論理 = Physical AI: 象限IVから象限Iへの垂直移動',
    ]
    for ev in evolutions:
        story.append(Paragraph(f'• {ev}', s['bullet']))

    story.append(PageBreak())

    # ==== 4. 地図2 ====
    story.append(Paragraph('4. 地図2: 知識の抽象度 × 応答レイテンシ', s['h1']))
    story.append(Paragraph(
        'AIがどの抽象レベルの知識を、どのタイミングで取得するかを可視化する地図。'
        '横軸に応答レイテンシ/知識の鮮度（事前準備型〜リアルタイム）、'
        '縦軸に知識の抽象度（具体的データ〜抽象的スキーマ）を配置する。', s['body']))

    map2_path = '/tmp/grounding_map2.png'
    generate_map2_chart(map2_path)
    story.append(Image(map2_path, width=W, height=W*0.72))
    story.append(Paragraph('図2: 知識の抽象度×応答レイテンシのスキャッタープロット（3ゾーンで分類）', s['caption']))

    story.append(Spacer(1, 3*mm))

    # 3ゾーン解説
    story.append(Paragraph('3つのゾーン', s['h2']))

    zones = [
        ('設計時知識ゾーン（上部）', '#ebf8ff', '#2b6cb0',
         'AIの「世界モデル」を事前に定義する手法群。変更頻度は低い（月〜年単位）。'
         'オントロジー、形式論理、タクソノミー、スキーマ、ファインチューニングが該当。'
         '安定した概念的基盤を提供するが、更新が困難。'),
        ('蓄積知識ゾーン（左下）', '#f0fff4', '#2f855a',
         'バッチ処理で蓄積される知識。ナレッジグラフ、エンベディングインデックス、'
         'メモリシステム、平ドキュメントが該当。中程度の更新頻度。'),
        ('動的知識ゾーン（右下）', '#fffff0', '#b7791f',
         'リアルタイムで取得される具体的な知識。RAG各種、ツール利用/MCP、Feature Store、'
         'マルチモーダル、デジタルツインが該当。秒〜分単位で変化する最新データ。'),
    ]
    for ztitle, bg_hex, color_hex, zdesc in zones:
        z_data = [[Paragraph(f'<b>{ztitle}</b>', ParagraphStyle(
            'ZH', fontName='IPAGothic', fontSize=10, leading=14,
            textColor=HexColor(color_hex)))],
            [Paragraph(zdesc, s['body_small'])]]
        z_table = Table(z_data, colWidths=[W])
        z_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(bg_hex)),
            ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2*mm),
            ('LEFTPADDING', (0, 0), (-1, -1), 3*mm),
            ('BOX', (0, 0), (-1, -1), 0.5, HexColor(color_hex)),
        ]))
        story.append(KeepTogether([z_table, Spacer(1, 2*mm)]))

    # 主要な読み取り
    story.append(Paragraph('主要な読み取りポイント', s['h2']))
    readings = [
        '<b>二層構造の必然性</b>: 効果的なAIシステムは設計時知識（枠組み）と実行時知識（具体的事実）の両方を組み合わせる',
        '<b>抽象度と鮮度のトレードオフ</b>: 左上（高抽象・低鮮度）と右下（低抽象・高鮮度）の対角線の両端を結ぶのが最適',
        '<b>ガードレールの特異性</b>: 設計時に定義された抽象原則を実行時にリアルタイム適用する唯一の手法',
        '<b>知識の「重力」</b>: 上方が「骨格」、下方が「肉付け」。骨格なき肉付けはハルシネーション、肉付けなき骨格は空虚',
    ]
    for r in readings:
        story.append(Paragraph(f'• {r}', s['bullet']))

    story.append(PageBreak())

    # ==== 5. 地図3 ====
    story.append(Paragraph('5. 地図3: データライフサイクル・パイプライン', s['h1']))
    story.append(Paragraph(
        '現実世界の知識がAIに到達するまでの8段階パイプラインと、各手法のカバー範囲を'
        'ヒートマップで可視化する。', s['body']))

    # パイプライン説明
    pipeline_stages = [
        ('①ソース', '物理現象・業務文書・API等の知識源'),
        ('②構造化', '概念定義・関係定義・型定義・ルール定義'),
        ('③格納・索引化', 'ベクトルDB・グラフDB・特徴量ストアへの永続化'),
        ('④検索・取得', 'クエリに応じた知識の検索とフィルタリング'),
        ('⑤コンテキスト注入', 'プロンプト組み立て・コンテキストエンジニアリング'),
        ('⑥生成・推論', 'LLM推論・ツール呼び出し・知識結合'),
        ('⑦出力制御', 'バリデーション・ガードレール・構造化出力'),
        ('⑧フィードバック', 'メモリ蓄積・モデル再訓練・品質改善'),
    ]
    pipe_data = [[Paragraph('<b>ステージ</b>', s['table_header']),
                   Paragraph('<b>処理内容</b>', s['table_header'])]]
    for stage, desc in pipeline_stages:
        pipe_data.append([
            Paragraph(stage, s['table_cell']),
            Paragraph(desc, s['table_cell']),
        ])
    pipe_table = Table(pipe_data, colWidths=[W*0.2, W*0.8])
    pipe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_HEADER),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 2*mm),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, C_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(pipe_table)
    story.append(Spacer(1, 4*mm))

    # ヒートマップ
    map3_path = '/tmp/grounding_map3.png'
    generate_map3_chart(map3_path)
    story.append(Image(map3_path, width=W, height=W*0.67))
    story.append(Paragraph('図3: 各手法のパイプラインカバレッジ（●がカバー範囲、左端の色帯がカテゴリ）', s['caption']))

    story.append(PageBreak())

    # 地図3の読み取り
    story.append(Paragraph('パイプライン図の読み取りポイント', s['h2']))
    pipe_insights = [
        '<b>「誰も全ステージをカバーしない」</b>: どの手法も単独ではライフサイクル全体をカバーできない。これがハイブリッドアプローチの必然性を示す。',
        '<b>上流専門（①②）</b>: オントロジー、タクソノミー、デジタルツイン — 知識の定義・構造化に特化',
        '<b>中流専門（③④⑤）</b>: RAG、エンベディング検索 — 知識の格納・検索・注入に特化',
        '<b>下流専門（⑥⑦）</b>: ガードレール、スキーマ検証 — 出力の制御・品質保証に特化',
        '<b>フィードバック不足</b>: ステージ⑧をカバーする手法が少ない。メモリシステムとFeature Storeがギャップを埋める',
        '<b>エージェンティックAI</b>: 特定ステージに属さず、パイプライン全体をオーケストレーションする存在',
    ]
    for ins in pipe_insights:
        story.append(Paragraph(f'• {ins}', s['bullet']))

    # フルスタック例
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('フルスタック・グラウンディングの構成例', s['h3']))
    stack_data = [
        [Paragraph('<b>ステージ</b>', s['table_header']),
         Paragraph('<b>担当手法</b>', s['table_header']),
         Paragraph('<b>役割</b>', s['table_header'])],
        ['②構造化', 'OWLオントロジー', '概念定義・ドメインモデル'],
        ['②③④', 'ナレッジグラフ', 'エンティティ・関係の格納と検索'],
        ['③④⑤', 'Graph RAG', 'グラフ走査による検索と注入'],
        ['⑦', 'ガードレール', '安全性・品質チェック'],
        ['⑤⑧', 'メモリシステム', '対話履歴の注入とフィードバック'],
    ]
    for i in range(1, len(stack_data)):
        stack_data[i] = [Paragraph(stack_data[i][j], s['table_cell']) for j in range(3)]
    stack_table = Table(stack_data, colWidths=[W*0.15, W*0.35, W*0.5])
    stack_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_HEADER),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 2*mm),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, C_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(stack_table)

    story.append(PageBreak())

    # ==== 6. 技術横断比較マトリクス ====
    story.append(Paragraph('6. 技術横断比較マトリクス', s['h1']))
    story.append(Paragraph(
        '全手法を5つの評価軸で横断的に比較する。', s['body']))

    # 比較表
    matrix_header = [
        Paragraph('<b>手法</b>', s['table_header']),
        Paragraph('<b>意味の\n明示度</b>', s['table_header']),
        Paragraph('<b>運用\n結合度</b>', s['table_header']),
        Paragraph('<b>スケーラ\nビリティ</b>', s['table_header']),
        Paragraph('<b>維持\nコスト</b>', s['table_header']),
        Paragraph('<b>知識の鮮度</b>', s['table_header']),
    ]
    matrix_data = [matrix_header]
    matrix_rows = [
        ('OWLオントロジー',      '★★★★★', '★★★★', '★★',   '高',   '手動更新'),
        ('ナレッジグラフ',       '★★★★',  '★★★★', '★★★',  '中〜高', 'パイプライン'),
        ('タクソノミー',         '★★★',   '★★★',  '★★★★', '低〜中', '手動レビュー'),
        ('スキーマ/データモデル', '★★★',   '★★★★', '★★★★', '中',   '設計時固定'),
        ('形式論理/ルール',      '★★★★★', '★★★★★','★★',   '高',   '手動更新'),
        ('RAG (Naive/Advanced)', '★★★',   '★★★',  '★★★★', '中',   'リアルタイム'),
        ('Graph RAG',           '★★★★',  '★★★★', '★★★',  '高',   'リアルタイム'),
        ('平ドキュメント',       '★★',    '★',    '★★',   '最低',  '手動更新'),
        ('エンベディング検索',    '★★★',   '★★★',  '★★★★', '中',   'インデックス更新'),
        ('構造化プロンプティング', '★★★',   '★',    '★★★',  '最低',  'モデル依存'),
        ('ツール利用/MCP',       '★★★★',  '★★★★★','★★★★', '中',   'リアルタイム'),
        ('デジタルツイン',       '★★★★',  '★★★★★','★★',   '最高',  'リアルタイム'),
        ('マルチモーダル',       '★★★',   '★★★★', '★★★',  '高',   'リアルタイム'),
        ('ファインチューニング',  '★',     '★★',   '★★',   '高',   '訓練時固定'),
        ('ガードレール',         '★★★',   '★★★★', '★★★',  '中',   'ルール更新'),
        ('セマンティックレイヤー', '★★★★',  '★★★★', '★★★★', '中',   'メタデータ更新'),
        ('Feature Store',       '★★★',   '★★★★', '★★★★', '中',   'パイプライン'),
        ('メモリシステム',       '★★',    '★★★',  '★★★',  '中',   '自己更新'),
    ]
    for row in matrix_rows:
        matrix_data.append([Paragraph(cell, s['table_cell']) for cell in row])

    col_w = [W*0.22, W*0.13, W*0.13, W*0.13, W*0.12, W*0.27]
    m_table = Table(matrix_data, colWidths=col_w)
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_HEADER),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('GRID', (0, 0), (-1, -1), 0.4, C_BORDER),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 1.5*mm),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, C_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(m_table)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '※ ★の数は度合いの高さを表す。維持コストは「高」ほどコストが大きい。', s['body_small']))

    story.append(PageBreak())

    # ==== 7. 手法選定ガイド ====
    story.append(Paragraph('7. 手法選定ガイドと進化の方向性', s['h1']))

    story.append(Paragraph('要件に応じた手法選定フロー', s['h2']))

    flow_data = [
        [Paragraph('<b>要件</b>', s['table_header']),
         Paragraph('<b>推奨手法</b>', s['table_header']),
         Paragraph('<b>理由</b>', s['table_header'])],
    ]
    flow_rows = [
        ('手軽に始めたい（PoC）',
         '平ドキュメント\n+ 構造化プロンプティング',
         'インフラ不要。テキスト編集だけで即開始可能'),
        ('大量文書からの質問応答',
         'RAG（Advanced）\n+ エンベディング検索',
         '動的検索で知識量の壁を突破。ハイブリッド検索で精度向上'),
        ('複雑な関連性の推論が必要',
         'Graph RAG\n+ ナレッジグラフ',
         'マルチホップ推論とエンティティ間の関係走査が可能'),
        ('リアルタイムの外部データが必要',
         'ツール利用/MCP\n（+ RAG）',
         'API経由の最新データ取得。MCPで統合を標準化'),
        ('規制遵守・安全性が最優先',
         'ガードレール\n+ 形式論理\n+ オントロジー',
         '多層的な出力制御と監査可能な意思決定過程'),
        ('物理世界との統合',
         'デジタルツイン\n+ マルチモーダル',
         'センサーデータと物理法則に基づくAI予測'),
        ('モデルの行動・スタイル変更',
         'ファインチューニング/RLHF\n（+ RAG）',
         'パラメータレベルでの行動制御。RAGと組み合わせが最適'),
    ]
    for req, rec, reason in flow_rows:
        flow_data.append([
            Paragraph(req, s['table_cell']),
            Paragraph(rec, s['table_cell']),
            Paragraph(reason, s['table_cell']),
        ])
    flow_table = Table(flow_data, colWidths=[W*0.25, W*0.3, W*0.45])
    flow_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_HEADER),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 2*mm),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, C_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(flow_table)

    story.append(Spacer(1, 5*mm))

    # 2025-2026トレンド
    story.append(Paragraph('2025-2026年の主要トレンド', s['h2']))
    trends = [
        ('<b>コンテキストエンジニアリングの台頭</b>: '
         '単なるRAGを超え、タスク・場面に応じた最適なコンテキストを動的に組み立てる技術が最重要に'),
        ('<b>エージェンティックAIの標準化</b>: '
         '全グラウンディング手法を統合するオーケストレーション層としてのAIエージェントが普及。'
         '85%のエンタープライズが2025年にAIエージェントをワークフローに組み込むと予測'),
        ('<b>MCPの業界標準確立</b>: '
         'Anthropic発のModel Context Protocolが2025年12月にLinux Foundation傘下に移管。'
         'AIとツール・データの接続における事実上の標準'),
        ('<b>ニューロシンボリックAIの実用化</b>: '
         'LLMの柔軟性と形式論理の厳密性を組み合わせるハイブリッドが主流に。'
         'Amazonが倉庫ロボットに適用済み'),
        ('<b>ガバナンスの設計時組み込み</b>: '
         'EU AI Act等への対応として、後付けではなく設計段階からガバナンスを組み込む動きが加速'),
        ('<b>RAGの「知識ランタイム」化</b>: '
         'RAGは単なる検索パターンから、検証・推論・アクセス制御・監査証跡を統合管理する'
         'オーケストレーション層へ進化'),
    ]
    for t in trends:
        story.append(Paragraph(f'• {t}', s['bullet']))

    story.append(Spacer(1, 8*mm))

    # 結語
    conclusion_data = [[Paragraph(
        '<b>結語</b>: グラウンディングの未来は、単一手法の選択ではなく、'
        '複数手法の多層的合成にある。本資料の3つの地図は、その合成設計の'
        '羅針盤として活用していただきたい。',
        ParagraphStyle('Conclusion', fontName='IPAPGothic', fontSize=10,
                      leading=16, textColor=C_PRIMARY))]]
    conc_table = Table(conclusion_data, colWidths=[W])
    conc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#ebf8ff')),
        ('BOX', (0, 0), (-1, -1), 1, C_SECONDARY),
        ('TOPPADDING', (0, 0), (-1, -1), 4*mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4*mm),
        ('LEFTPADDING', (0, 0), (-1, -1), 4*mm),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4*mm),
    ]))
    story.append(conc_table)

    # ==== ビルド ====
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF generated: {output_path}")


# ==============================================================================
# メイン
# ==============================================================================
if __name__ == '__main__':
    output = '/home/user/grounding-research/grounding-techniques-map.pdf'
    build_pdf(output)
