#!/usr/bin/env python3
"""
AIグラウンディング手法マップ — PDF解説資料 v2

再設計方針:
- 「機能が異なるもの」を同列に比較しない
- 4層モデルで手法を整理し、層内でのみ比較する
- 各手法の原理（なぜ効くのか）を明記する
- 層間の合成パターンを示す
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

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
fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# カラーパレット
# ==============================================================================
C_PRIMARY    = HexColor('#1a365d')
C_SECONDARY  = HexColor('#2b6cb0')
C_ACCENT     = HexColor('#e53e3e')
C_BG_LIGHT   = HexColor('#f7fafc')
C_BG_HEADER  = HexColor('#2d3748')
C_TEXT       = HexColor('#1a202c')
C_TEXT_LIGHT = HexColor('#4a5568')
C_BORDER     = HexColor('#e2e8f0')

# 層カラー
L1_COLOR = '#2f855a'  # 緑: 知識の構造化
L2_COLOR = '#2b6cb0'  # 青: 知識の配送
L3_COLOR = '#d69e2e'  # 黄: モデルへの統合
L4_COLOR = '#9b2c2c'  # 赤: 出力の保証

# ==============================================================================
# スタイル定義
# ==============================================================================
def make_styles():
    s = {}
    s['title'] = ParagraphStyle(
        'Title', fontName='IPAGothic', fontSize=26, leading=34,
        textColor=white, alignment=TA_CENTER, spaceAfter=6*mm)
    s['subtitle'] = ParagraphStyle(
        'Subtitle', fontName='IPAPGothic', fontSize=13, leading=20,
        textColor=HexColor('#a0aec0'), alignment=TA_CENTER, spaceAfter=4*mm)
    s['h1'] = ParagraphStyle(
        'H1', fontName='IPAGothic', fontSize=18, leading=26,
        textColor=C_PRIMARY, spaceBefore=6*mm, spaceAfter=4*mm)
    s['h2'] = ParagraphStyle(
        'H2', fontName='IPAGothic', fontSize=14, leading=20,
        textColor=C_SECONDARY, spaceBefore=5*mm, spaceAfter=3*mm)
    s['h3'] = ParagraphStyle(
        'H3', fontName='IPAGothic', fontSize=11, leading=16,
        textColor=C_TEXT, spaceBefore=3*mm, spaceAfter=2*mm)
    s['body'] = ParagraphStyle(
        'Body', fontName='IPAPGothic', fontSize=9.5, leading=16,
        textColor=C_TEXT, spaceAfter=2*mm, alignment=TA_JUSTIFY)
    s['body_small'] = ParagraphStyle(
        'BodySmall', fontName='IPAPGothic', fontSize=8.5, leading=14,
        textColor=C_TEXT_LIGHT, spaceAfter=1.5*mm)
    s['bullet'] = ParagraphStyle(
        'Bullet', fontName='IPAPGothic', fontSize=9.5, leading=15,
        textColor=C_TEXT, leftIndent=8*mm, bulletIndent=3*mm, spaceAfter=1*mm)
    s['caption'] = ParagraphStyle(
        'Caption', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT_LIGHT, alignment=TA_CENTER, spaceAfter=2*mm)
    s['toc'] = ParagraphStyle(
        'TOC', fontName='IPAPGothic', fontSize=11, leading=20,
        textColor=C_PRIMARY, leftIndent=5*mm)
    s['th'] = ParagraphStyle(
        'TH', fontName='IPAGothic', fontSize=8.5, leading=12,
        textColor=white, alignment=TA_CENTER)
    s['tc'] = ParagraphStyle(
        'TC', fontName='IPAPGothic', fontSize=8, leading=12, textColor=C_TEXT)
    s['tcc'] = ParagraphStyle(
        'TCC', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT, alignment=TA_CENTER)
    return s

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('IPAPGothic', 8)
    canvas.setFillColor(C_TEXT_LIGHT)
    pn = canvas.getPageNumber()
    if pn > 1:
        canvas.drawCentredString(A4[0]/2, 12*mm, f"- {pn} -")
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(20*mm, A4[1]-15*mm, A4[0]-20*mm, A4[1]-15*mm)
        canvas.line(20*mm, 18*mm, A4[0]-20*mm, 18*mm)
    canvas.restoreState()


# ==============================================================================
# 図1: 4層モデル概念図
# ==============================================================================
def generate_layer_model(output_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    layers = [
        (0.8, 6.0, 10.4, 1.3, L4_COLOR, '第4層: 出力の保証',
         'ガードレール / Constitutional AI / スキーマバリデーション',
         '「AIの出力を制約・検証する」'),
        (0.8, 4.3, 10.4, 1.3, L3_COLOR, '第3層: モデルへの統合',
         'ファインチューニング / RLHF / 構造化プロンプティング',
         '「知識をモデルの内部に統合する」'),
        (0.8, 2.6, 10.4, 1.3, L2_COLOR, '第2層: 知識の配送',
         'RAG各種 / ツール利用(MCP) / エンベディング検索 / メモリシステム',
         '「知識をモデルに届ける」'),
        (0.8, 0.9, 10.4, 1.3, L1_COLOR, '第1層: 知識の構造化',
         'オントロジー / ナレッジグラフ / タクソノミー / スキーマ',
         '「現実世界の知識を構造化して表現する」'),
    ]

    for x, y, w, h, color, title, techs, principle in layers:
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.35, title, fontsize=13,
                color='white', fontweight='bold', va='top')
        ax.text(x + 0.3, y + h - 0.75, principle, fontsize=9,
                color='#e2e8f0', va='top', style='italic')
        ax.text(x + 0.3, y + 0.2, techs, fontsize=8.5,
                color='#e2e8f0', va='bottom')

    # 矢印（層間の流れ）
    for y_start, y_end in [(2.25, 2.6), (3.9, 4.3), (5.6, 6.0)]:
        ax.annotate('', xy=(6, y_end), xytext=(6, y_start),
                    arrowprops=dict(arrowstyle='->', color='#a0aec0',
                                   lw=2.5, alpha=0.7))

    # 左に注釈
    ax.text(0.15, 1.55, '知識\nソース', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')
    ax.text(0.15, 3.25, '配送\n経路', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')
    ax.text(0.15, 4.95, '統合\n方式', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')
    ax.text(0.15, 6.65, '品質\n保証', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')

    # 右に共通問い
    ax.text(11.8, 1.55, '何を\n知るか', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')
    ax.text(11.8, 3.25, 'どう\n届けるか', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')
    ax.text(11.8, 4.95, 'どう\n入れるか', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')
    ax.text(11.8, 6.65, 'どう\n守るか', fontsize=9, color='#718096',
            ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

# ==============================================================================
# 図2: 層1 — 知識の構造化手法の比較（表現力 × 構築コスト）
# ==============================================================================
def generate_layer1_chart(output_path):
    techniques = [
        ("平ドキュメント\n(Markdown等)", 1.0, 1.0, 60),
        ("タクソノミー/\n統制語彙", 2.0, 1.8, 80),
        ("スキーマ/\nデータモデル", 2.5, 2.5, 90),
        ("ナレッジグラフ", 3.5, 3.5, 110),
        ("OWLオントロジー", 4.8, 4.5, 110),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # 背景ゾーン
    from matplotlib.patches import FancyBboxPatch
    r1 = FancyBboxPatch((0.5, 0.5), 2.0, 2.0, boxstyle="round,pad=0.1",
                        facecolor='#f0fff4', edgecolor='#9ae6b4',
                        linewidth=1, alpha=0.4)
    ax.add_patch(r1)
    ax.text(1.5, 2.3, '軽量ゾーン', fontsize=9, ha='center',
            color='#2f855a', fontweight='bold', alpha=0.6)

    r2 = FancyBboxPatch((2.7, 2.7), 2.8, 2.3, boxstyle="round,pad=0.1",
                        facecolor='#ebf8ff', edgecolor='#90cdf4',
                        linewidth=1, alpha=0.4)
    ax.add_patch(r2)
    ax.text(4.1, 4.8, '形式ゾーン', fontsize=9, ha='center',
            color='#2b6cb0', fontweight='bold', alpha=0.6)

    # 進化の矢印
    coords = [(t[1], t[2]) for t in techniques]
    for i in range(len(coords)-1):
        ax.annotate('', xy=coords[i+1], xytext=coords[i],
                    arrowprops=dict(arrowstyle='->', color='#a0aec0',
                                   lw=1.5, alpha=0.5,
                                   connectionstyle='arc3,rad=0.1'))

    for name, x, y, size in techniques:
        ax.scatter(x, y, c=L1_COLOR, s=size*1.5, zorder=5,
                   edgecolors='white', linewidth=1.5, alpha=0.9)
        ax.annotate(name, (x, y), fontsize=8.5, color='#2d3748',
                    xytext=(x+0.15, y+0.15), fontweight='bold')

    ax.text(3.0, 0.7, '共通原理: 現実世界の構造を機械可読な形式で記述する',
            fontsize=10, ha='center', color='#2d3748',
            fontweight='bold', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7fafc',
                     edgecolor='#e2e8f0'))

    ax.set_xlim(0.3, 5.5)
    ax.set_ylim(0.3, 5.3)
    ax.set_xlabel('知識の表現力（概念・関係・制約の記述能力）  →', fontsize=10, labelpad=8)
    ax.set_ylabel('構築・維持の労力  →', fontsize=10, labelpad=8)
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ==============================================================================
# 図3: 層2 — 知識の配送手法の比較（検索精度 × レイテンシ）
# ==============================================================================
def generate_layer2_chart(output_path):
    techniques = [
        ("平ドキュメント注入", 1.0, 1.0, 60),
        ("Naive RAG", 2.0, 2.5, 80),
        ("エンベディング検索", 2.5, 2.0, 80),
        ("Advanced RAG\n(ハイブリッド検索)", 3.5, 3.0, 100),
        ("ツール利用/MCP", 4.5, 2.0, 100),
        ("Graph RAG", 4.0, 4.5, 110),
        ("メモリシステム\n(MemGPT等)", 3.0, 3.5, 90),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # RAG進化の矢印
    rag_indices = [0, 1, 3, 5]
    rag_coords = [(techniques[i][1], techniques[i][2]) for i in rag_indices]
    for i in range(len(rag_coords)-1):
        ax.annotate('', xy=rag_coords[i+1], xytext=rag_coords[i],
                    arrowprops=dict(arrowstyle='->', color='#718096',
                                   lw=1.5, alpha=0.5,
                                   connectionstyle='arc3,rad=0.15'))

    # ラベル
    ax.text(2.5, 4.8, 'RAG進化の系譜', fontsize=9, ha='center',
            color='#718096', fontweight='bold', alpha=0.6)

    for name, x, y, size in techniques:
        is_rag = 'RAG' in name or 'Naive' in name or '平ドキュメント' in name
        color = '#2b6cb0' if is_rag else '#d69e2e'
        ax.scatter(x, y, c=color, s=size*1.5, zorder=5,
                   edgecolors='white', linewidth=1.5, alpha=0.9)
        ox, oy = 0.12, 0.15
        if 'ツール' in name:
            ox, oy = 0.15, -0.2
        elif 'メモリ' in name:
            ox, oy = 0.15, -0.25
        ax.annotate(name, (x, y), fontsize=8, color='#2d3748',
                    xytext=(x+ox, y+oy), fontweight='bold')

    ax.text(3.0, 0.6, '共通原理: 外部の知識をモデルのコンテキストに配送する',
            fontsize=10, ha='center', color='#2d3748',
            fontweight='bold', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7fafc',
                     edgecolor='#e2e8f0'))

    # 凡例
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#2b6cb0', label='テキスト検索系（RAGの系譜）'),
              Patch(facecolor='#d69e2e', label='外部接続系（API/メモリ）')]
    ax.legend(handles=legend, loc='upper left', fontsize=8,
              framealpha=0.9, edgecolor='#e2e8f0')

    ax.set_xlim(0.3, 5.5)
    ax.set_ylim(0.3, 5.3)
    ax.set_xlabel('知識の鮮度・更新頻度  →（リアルタイム）', fontsize=10, labelpad=8)
    ax.set_ylabel('検索精度・文脈理解度  →', fontsize=10, labelpad=8)
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ==============================================================================
# 図4: 層間合成パターン図
# ==============================================================================
def generate_composition_chart(output_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # パターン1: 基本構成
    ax.text(2.0, 7.5, 'パターンA: 基本構成（PoC向け）', fontsize=11,
            ha='center', fontweight='bold', color='#2d3748')
    boxes_a = [
        (0.5, 5.8, 3.0, 0.7, L2_COLOR, 'L2: 平ドキュメント注入'),
        (0.5, 5.0, 3.0, 0.7, L3_COLOR, 'L3: 構造化プロンプティング'),
    ]
    for x, y, w, h, c, t in boxes_a:
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=c, edgecolor='white', lw=1.5, alpha=0.8)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, t, fontsize=8, ha='center', va='center',
                color='white', fontweight='bold')

    # パターン2: 標準構成
    ax.text(6.0, 7.5, 'パターンB: 標準構成', fontsize=11,
            ha='center', fontweight='bold', color='#2d3748')
    boxes_b = [
        (4.5, 6.4, 3.0, 0.6, L1_COLOR, 'L1: スキーマ定義'),
        (4.5, 5.7, 3.0, 0.6, L2_COLOR, 'L2: Advanced RAG'),
        (4.5, 5.0, 3.0, 0.6, L3_COLOR, 'L3: 構造化プロンプティング'),
        (4.5, 4.3, 3.0, 0.6, L4_COLOR, 'L4: スキーマバリデーション'),
    ]
    for x, y, w, h, c, t in boxes_b:
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=c, edgecolor='white', lw=1.5, alpha=0.8)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, t, fontsize=8, ha='center', va='center',
                color='white', fontweight='bold')

    # パターン3: エンタープライズ構成
    ax.text(10.0, 7.5, 'パターンC: エンタープライズ', fontsize=11,
            ha='center', fontweight='bold', color='#2d3748')
    boxes_c = [
        (8.5, 6.4, 3.0, 0.6, L1_COLOR, 'L1: KG + オントロジー'),
        (8.5, 5.7, 3.0, 0.6, L2_COLOR, 'L2: Graph RAG + MCP'),
        (8.5, 5.0, 3.0, 0.6, L3_COLOR, 'L3: FT + プロンプト'),
        (8.5, 4.3, 3.0, 0.6, L4_COLOR, 'L4: ガードレール + 監査'),
    ]
    for x, y, w, h, c, t in boxes_c:
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=c, edgecolor='white', lw=1.5, alpha=0.8)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, t, fontsize=8, ha='center', va='center',
                color='white', fontweight='bold')

    # 下半分: なぜ層を分けるのか
    ax.text(6.0, 3.5, '層を分ける理由: 各層の手法は異なる問いに答えている', fontsize=12,
            ha='center', fontweight='bold', color='#1a365d')

    questions = [
        (1.5, 2.5, L1_COLOR, '第1層', '知識をどう表現するか？\n→ 表現力と構築コストのトレードオフ'),
        (4.5, 2.5, L2_COLOR, '第2層', '知識をどう届けるか？\n→ 鮮度と検索精度のトレードオフ'),
        (7.5, 2.5, L3_COLOR, '第3層', '知識をどう入れるか？\n→ 柔軟性と永続性のトレードオフ'),
        (10.5, 2.5, L4_COLOR, '第4層', '出力をどう守るか？\n→ 安全性と応答速度のトレードオフ'),
    ]
    for x, y, c, title, desc in questions:
        r = FancyBboxPatch((x-1.2, y-1.0), 2.4, 2.0, boxstyle="round,pad=0.08",
                          facecolor=c, edgecolor='white', lw=1.5, alpha=0.15)
        ax.add_patch(r)
        ax.text(x, y+0.6, title, fontsize=10, ha='center', va='center',
                color=c, fontweight='bold')
        ax.text(x, y-0.15, desc, fontsize=7.5, ha='center', va='center',
                color='#4a5568')

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
        title='AIグラウンディング手法 原理と層構造',
        author='Grounding Research Team')

    story = []
    W = A4[0] - 40*mm

    # ============ 表紙 ============
    story.append(Spacer(1, 45*mm))
    title_data = [
        [Paragraph('AIグラウンディング手法の原理と層構造', s['title'])],
        [Paragraph('— 何が比較可能で、何が合成すべきかを理解する —', s['subtitle'])],
        [Spacer(1, 8*mm)],
        [Paragraph('各手法の「なぜ効くのか」を原理から解説し、<br/>機能の異なる手法を4層モデルで整理する', s['subtitle'])],
    ]
    tt = Table(title_data, colWidths=[W])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_HEADER),
        ('TOPPADDING', (0,0), (-1,-1), 8*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 5*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 5*mm),
    ]))
    story.append(tt)
    story.append(Spacer(1, 15*mm))
    meta = ParagraphStyle('Meta', fontName='IPAPGothic', fontSize=10,
                          leading=16, textColor=C_TEXT_LIGHT, alignment=TA_CENTER)
    story.append(Paragraph('2026年3月', meta))
    story.append(PageBreak())

    # ============ 目次 ============
    story.append(Paragraph('目次', s['h1']))
    toc = [
        '1. 旧版の問題: なぜガードレールとオントロジーを比較してはいけないのか',
        '2. 4層モデル: グラウンディングの層構造',
        '3. 第1層 — 知識の構造化（原理と手法比較）',
        '4. 第2層 — 知識の配送（原理と手法比較）',
        '5. 第3層 — モデルへの統合（原理と手法比較）',
        '6. 第4層 — 出力の保証（原理と手法比較）',
        '7. 層間合成パターン: 手法の組み合わせ方',
        '8. 技術原理の類似性マトリクス',
    ]
    for item in toc:
        story.append(Paragraph(item, s['toc']))
    story.append(PageBreak())

    # ============ 1. 問題提起 ============
    story.append(Paragraph('1. 旧版の問題: なぜガードレールとオントロジーを比較してはいけないのか', s['h1']))
    story.append(Paragraph(
        '旧版の資料では、オントロジー、RAG、ガードレール、ファインチューニング等を'
        '同じ座標系上にプロットし比較していた。しかしこれは根本的に問題がある。'
        'これらは<b>同じ機能を果たす代替手段ではなく、異なる機能を担う補完的な層</b>だからだ。', s['body']))

    prob_data = [
        [Paragraph('<b>手法</b>', s['th']),
         Paragraph('<b>やっていること</b>', s['th']),
         Paragraph('<b>機能カテゴリ</b>', s['th'])],
        [Paragraph('オントロジー', s['tc']),
         Paragraph('ドメインの概念・関係を定義する', s['tc']),
         Paragraph('知識の構造化', s['tc'])],
        [Paragraph('RAG', s['tc']),
         Paragraph('外部テキストを検索して注入する', s['tc']),
         Paragraph('知識の配送', s['tc'])],
        [Paragraph('ファインチューニング', s['tc']),
         Paragraph('モデルの重みを更新する', s['tc']),
         Paragraph('モデルへの統合', s['tc'])],
        [Paragraph('ガードレール', s['tc']),
         Paragraph('出力を検証・フィルタリングする', s['tc']),
         Paragraph('出力の保証', s['tc'])],
    ]
    pt = Table(prob_data, colWidths=[W*0.22, W*0.43, W*0.35])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 2*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pt)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        'オントロジーとRAGを「どちらが優れているか」と比較することは、'
        '建物の設計図と配送トラックを「どちらが優れているか」と比較するのと同じだ。'
        '設計図（構造化）は建物の形を定義し、配送トラック（RAG）は建材を届ける。'
        'それぞれ異なる問いに答えている。', s['body']))
    story.append(Paragraph(
        '<b>比較が意味を持つのは、同じ問いに対する異なる答えの間だけ</b>である。'
        '例:「知識をどう構造化するか」という問いに対して、オントロジー vs ナレッジグラフ vs タクソノミーの比較は有意義だ。', s['body']))

    # 比較可能/不可能の図解
    comp_data = [
        [Paragraph('<b>比較</b>', s['th']),
         Paragraph('<b>意味があるか</b>', s['th']),
         Paragraph('<b>理由</b>', s['th'])],
        [Paragraph('オントロジー vs ナレッジグラフ', s['tc']),
         Paragraph('意味がある', s['tcc']),
         Paragraph('同じ問い「知識をどう構造化するか」への異なる答え', s['tc'])],
        [Paragraph('Naive RAG vs Advanced RAG', s['tc']),
         Paragraph('意味がある', s['tcc']),
         Paragraph('同じ問い「知識をどう届けるか」への異なる答え', s['tc'])],
        [Paragraph('ファインチューニング vs プロンプティング', s['tc']),
         Paragraph('意味がある', s['tcc']),
         Paragraph('同じ問い「知識をどうモデルに入れるか」への異なる答え', s['tc'])],
        [Paragraph('オントロジー vs ガードレール', s['tc']),
         Paragraph('意味がない', s['tcc']),
         Paragraph('答えている問いが異なる（構造化 vs 出力制御）', s['tc'])],
        [Paragraph('RAG vs ファインチューニング', s['tc']),
         Paragraph('条件付き', s['tcc']),
         Paragraph('「知識をモデルに入れる」意味では比較可能だが層が異なる', s['tc'])],
    ]
    ct = Table(comp_data, colWidths=[W*0.30, W*0.15, W*0.55])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 2*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('BACKGROUND', (1,4), (1,4), HexColor('#FED7D7')),
        ('BACKGROUND', (1,5), (1,5), HexColor('#FEFCBF')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(ct)

    story.append(PageBreak())

    # ============ 2. 4層モデル ============
    story.append(Paragraph('2. 4層モデル: グラウンディングの層構造', s['h1']))
    story.append(Paragraph(
        'グラウンディング手法を機能に基づき4つの層に整理する。各層は異なる問いに答えており、'
        '同一層内の手法同士が比較対象、異なる層の手法同士が合成対象となる。', s['body']))

    layer_path = '/tmp/grounding_layers.png'
    generate_layer_model(layer_path)
    story.append(Image(layer_path, width=W, height=W*0.58))
    story.append(Paragraph('図1: グラウンディングの4層モデル — 各層は異なる問いに答える', s['caption']))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('各層の定義', s['h2']))

    layer_defs = [
        ('第1層: 知識の構造化', L1_COLOR,
         '「現実世界の知識を<b>どう表現する</b>か」に答える層。',
         '知識を人間とAIの両方が理解できる構造に変換する。'
         '表現力が高いほど推論が可能になるが、構築・維持のコストが上がる。',
         'オントロジー、ナレッジグラフ、タクソノミー、スキーマ、平ドキュメント'),
        ('第2層: 知識の配送', L2_COLOR,
         '「構造化された知識を<b>どうモデルに届ける</b>か」に答える層。',
         '第1層で構造化された知識（またはそのまま）を検索・取得し、'
         'モデルが利用できる形で配送する。鮮度と精度のバランスが核心。',
         'RAG各種、エンベディング検索、ツール利用/MCP、メモリシステム'),
        ('第3層: モデルへの統合', L3_COLOR,
         '「知識を<b>どうモデルの内部に統合する</b>か」に答える層。',
         'コンテキスト注入（一時的）とパラメータ更新（永続的）の2系統がある。'
         '柔軟性と永続性のトレードオフが核心。',
         'ファインチューニング/RLHF、構造化プロンプティング、マルチモーダル入力'),
        ('第4層: 出力の保証', L4_COLOR,
         '「AIの出力を<b>どう制約・検証する</b>か」に答える層。',
         'グラウンディングの最終防衛線。出力が事実・ルール・安全性基準に'
         '適合しているかを検証する。これは知識の提供ではなく品質保証。',
         'ガードレール/Constitutional AI、スキーマバリデーション、形式論理/ルールエンジン'),
    ]
    for title, color, question, desc, techs in layer_defs:
        ld = [
            [Paragraph(f'<b>{title}</b>', ParagraphStyle(
                'LH', fontName='IPAGothic', fontSize=10.5, leading=15, textColor=white))],
            [Paragraph(question, ParagraphStyle(
                'LQ', fontName='IPAPGothic', fontSize=9.5, leading=15, textColor=C_TEXT))],
            [Paragraph(desc, s['body_small'])],
            [Paragraph(f'<b>手法:</b> {techs}', s['body_small'])],
        ]
        lt = Table(ld, colWidths=[W])
        lt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), HexColor(color)),
            ('BACKGROUND', (0,1), (0,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 2*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
            ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([lt, Spacer(1, 2*mm)]))

    story.append(PageBreak())

    # ============ 3. 第1層 — 知識の構造化 ============
    story.append(Paragraph('3. 第1層 — 知識の構造化: 原理と手法比較', s['h1']))
    story.append(Paragraph(
        'この層の手法は「現実世界の構造を機械可読な形式で記述する」という共通原理を持つ。'
        '違いは<b>表現力の深さ</b>にある。', s['body']))

    # 各手法の原理
    l1_techs = [
        ('平ドキュメント（Markdown/PDF）',
         '非構造化テキストをそのまま知識源として使う。',
         '自然言語は人間にとって最も表現力が高い形式である。LLMは自然言語の理解に長けて'
         'いるため、構造化せずともある程度の知識伝達が可能。',
         '構造がないため、同じ文書でも解釈にブレが生じる。大規模になると検索精度が落ちる。'
         'CLAUDE.md等のシステムプロンプトは実質このアプローチ。'),
        ('タクソノミー / 統制語彙',
         '概念を階層的な「is-a」関係で分類する。',
         '人間の認知は分類（カテゴライゼーション）を基盤とする。'
         'AIも語彙が統制されることで同義語・多義語の混乱がなくなり、'
         '検索精度と推論の一貫性が向上する。',
         '表現できるのは上下関係のみ。「AはBの一種」以外の関係（原因、構成要素等）は表現不可。'),
        ('スキーマ / データモデル',
         'JSON Schema、SQL DDL等でデータの構造・型・制約を定義する。',
         'データの「形」を厳密に定義することで、AIの入出力を予測可能にする。'
         '構造化出力（Structured Output）の基盤であり、'
         'システム間の契約（API Contract）として機能する。',
         '個々のデータの形は定義できるが、概念間の意味的関係は表現できない。'
         '「顧客」と「注文」の関係はFK制約で表現するが、意味的豊かさはない。'),
        ('ナレッジグラフ',
         'エンティティ（ノード）と関係（エッジ）のグラフ構造で知識を表現する。',
         'グラフ構造は多対多の関係を自然に表現でき、'
         '「AがBの原因で、BがCを構成する」のようなマルチホップ推論が可能。'
         'トリプル（主語-述語-目的語）の集積が知識の網を形成する。',
         '任意の関係を定義できるが、関係の意味は暗黙的。'
         '「AはBに関連する」の「関連」の厳密な意味は保証されない。'),
        ('OWLオントロジー',
         '記述論理（Description Logic）に基づき、概念・関係・公理を厳密に定義する。',
         '形式論理に基づくため推論器（Reasoner）による自動導出が可能。'
         '例:「哺乳類は脊椎動物である」「クジラは哺乳類である」→「クジラは脊椎動物である」'
         'を自動推論。矛盾検出・分類の自動化も可能。',
         '最も表現力が高いが、構築にOWL/RDFの専門知識が必須。'
         'ドメインエキスパートとオントロジストの協業が必要で、コストが最大。'),
    ]

    for name, what, why, tradeoff in l1_techs:
        td = [
            [Paragraph(f'<b>{name}</b>', ParagraphStyle(
                'TN', fontName='IPAGothic', fontSize=9.5, leading=14, textColor=white))],
            [Paragraph(f'<b>何をするか:</b> {what}', s['body_small'])],
            [Paragraph(f'<b>なぜ効くのか（原理）:</b> {why}', s['body_small'])],
            [Paragraph(f'<b>限界・トレードオフ:</b> {tradeoff}', s['body_small'])],
        ]
        tt = Table(td, colWidths=[W])
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), HexColor(L1_COLOR)),
            ('BACKGROUND', (0,1), (0,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([tt, Spacer(1, 1.5*mm)]))

    story.append(Spacer(1, 2*mm))
    l1_path = '/tmp/grounding_l1.png'
    generate_layer1_chart(l1_path)
    story.append(Image(l1_path, width=W*0.9, height=W*0.58))
    story.append(Paragraph('図2: 第1層の手法比較 — 表現力と構築コストのトレードオフ（矢印は進化方向）', s['caption']))

    story.append(Paragraph(
        '<b>類似性の核心:</b> すべて「現実をモデル化する」が、モデルの精密さが異なる。'
        '平ドキュメントは自然言語という曖昧な表現、タクソノミーは階層のみ、スキーマは型と制約、'
        'ナレッジグラフは関係のネットワーク、オントロジーは論理的公理。'
        '表現力の階段を登るほど機械推論が可能になるが、人間の負荷も上がる。', s['body']))

    story.append(PageBreak())

    # ============ 4. 第2層 — 知識の配送 ============
    story.append(Paragraph('4. 第2層 — 知識の配送: 原理と手法比較', s['h1']))
    story.append(Paragraph(
        'この層の手法は「外部の知識をモデルのコンテキストに配送する」という共通原理を持つ。'
        '違いは<b>配送のメカニズム</b>にある。', s['body']))

    l2_techs = [
        ('Naive RAG',
         'クエリで文書を検索し、上位結果をそのままプロンプトに追加する。',
         '「最も関連性の高い情報を見つけてモデルに渡す」という最もシンプルな実装。'
         'ベクトル類似度がクエリと文書の意味的近さの近似として機能する。',
         'チャンク分割の品質に依存。文脈の喪失、無関係な情報の混入が頻発。'),
        ('Advanced RAG（ハイブリッド検索）',
         'クエリ変換、リランキング、ハイブリッド検索（密+疎）を組み合わせる。',
         'Naive RAGの各ステップを個別に最適化する。'
         'クエリ拡張で意図を明確化し、リランカーで精度を向上、'
         'ハイブリッド検索でキーワード一致と意味検索を両立する。',
         'パイプラインが複雑化し、レイテンシが増大。各ステップのチューニングが必要。'),
        ('Graph RAG',
         'ナレッジグラフの構造を利用して、関連エンティティを辿りながら情報を収集する。',
         'グラフ走査により「AがBに影響し、BがCを引き起こす」のような'
         '多段階の推論チェーンを構築できる。テキスト検索では見つからない'
         '構造的な関連性を発見する。',
         '第1層のナレッジグラフ構築が前提。グラフの品質が直接精度に影響。'),
        ('エンベディング検索（ベクトル検索）',
         'テキストを密ベクトルに変換し、ベクトル空間上の近傍検索で類似内容を取得する。',
         '意味的に類似したテキストはベクトル空間上で近い位置に配置される'
         '（分布仮説）。これにより、キーワードが一致しなくても'
         '意味的に関連する情報を発見できる。',
         '「意味が近い」と「回答に役立つ」は必ずしも一致しない。'),
        ('ツール利用 / MCP',
         'LLMがAPI呼び出しを介して外部システムにリアルタイムアクセスする。',
         'モデルは「どのツールを、どの引数で呼ぶか」を判断し、'
         '結果をコンテキストに統合する。MCPにより接続の標準化が実現。'
         '検索ではなく実行（API呼び出し、計算、データ取得）が本質。',
         'ツールの信頼性に依存。API呼び出しのレイテンシとコスト。安全性の確保が課題。'),
        ('メモリシステム（MemGPT/Letta等）',
         'LLMの固定コンテキストウィンドウを超えた持続的記憶を提供する。',
         'OSの仮想メモリに着想を得た設計。'
         'メインメモリ（コンテキストウィンドウ）とアーカイブ（外部ストレージ）を'
         'LLM自身が管理し、必要な記憶をページイン/アウトする。',
         '何を記憶し何を忘れるかの判断をLLMに委ねるリスク。実装の複雑さ。'),
    ]

    for name, what, why, tradeoff in l2_techs:
        td = [
            [Paragraph(f'<b>{name}</b>', ParagraphStyle(
                'TN2', fontName='IPAGothic', fontSize=9.5, leading=14, textColor=white))],
            [Paragraph(f'<b>何をするか:</b> {what}', s['body_small'])],
            [Paragraph(f'<b>なぜ効くのか（原理）:</b> {why}', s['body_small'])],
            [Paragraph(f'<b>限界・トレードオフ:</b> {tradeoff}', s['body_small'])],
        ]
        tt = Table(td, colWidths=[W])
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), HexColor(L2_COLOR)),
            ('BACKGROUND', (0,1), (0,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([tt, Spacer(1, 1.5*mm)]))

    story.append(Spacer(1, 2*mm))
    l2_path = '/tmp/grounding_l2.png'
    generate_layer2_chart(l2_path)
    story.append(Image(l2_path, width=W*0.9, height=W*0.58))
    story.append(Paragraph('図3: 第2層の手法比較 — 鮮度×精度（矢印はRAG進化の系譜）', s['caption']))

    story.append(Paragraph(
        '<b>類似性の核心:</b> すべて「外部知識をモデルに届ける」が、配送メカニズムが異なる。'
        'RAG系は「事前に蓄えた文書を検索」、ツール利用は「リアルタイムにAPIを呼ぶ」、'
        'メモリは「過去の対話を呼び戻す」。検索 vs 実行 vs 記憶想起という3つのアプローチ。', s['body']))

    story.append(PageBreak())

    # ============ 5. 第3層 — モデルへの統合 ============
    story.append(Paragraph('5. 第3層 — モデルへの統合: 原理と手法比較', s['h1']))
    story.append(Paragraph(
        'この層の手法は「知識をモデルの内部に統合する」という共通原理を持つ。'
        '核心的な違いは<b>一時的（コンテキスト注入）か永続的（パラメータ更新）か</b>にある。', s['body']))

    l3_techs = [
        ('構造化プロンプティング（CoT、Few-shot等）',
         'プロンプト設計により、モデルの推論パターンを誘導する。',
         'LLMはin-context learningの能力を持つ。'
         '適切な例示（Few-shot）や推論ステップの明示（Chain-of-Thought）により、'
         'パラメータを変更せずにモデルの振る舞いを変えられる。'
         '知識というよりも「考え方の型」を注入する。',
         '一時的。コンテキストウィンドウの消費。プロンプトの品質に強く依存。'
         'モデルが変わると再設計が必要。'),
        ('ファインチューニング / RLHF',
         'ドメイン固有のデータでモデルのパラメータ（重み）を更新する。',
         '勾配降下法によりモデルの内部表現を変更し、'
         '特定タスク・ドメインへの適応を実現する。'
         'RLHFは人間のフィードバックを報酬関数に変換し、'
         'モデルの行動選好を調整する。知識がパラメータに「焼き込まれる」。',
         '永続的だが更新が困難（再訓練が必要）。大量の学習データと計算資源が必要。'
         '新しい知識との矛盾（catastrophic forgetting）。事実の埋め込みよりも'
         'スタイル・トーン・行動パターンの制御に向く。'),
        ('マルチモーダル入力',
         '画像・音声・動画等の非テキストデータをモデルに入力する。',
         '人間の認知は多感覚統合で成り立つ。'
         'テキストだけでは伝えにくい空間的・視覚的情報を直接入力することで、'
         'モデルの理解を物理世界に固定する。設計図、写真、グラフなどが典型。',
         '一時的（コンテキストウィンドウ内）。モダリティ間の解釈ギャップ。'
         'テキストより大きなトークン消費。'),
    ]

    for name, what, why, tradeoff in l3_techs:
        td = [
            [Paragraph(f'<b>{name}</b>', ParagraphStyle(
                'TN3', fontName='IPAGothic', fontSize=9.5, leading=14, textColor=white))],
            [Paragraph(f'<b>何をするか:</b> {what}', s['body_small'])],
            [Paragraph(f'<b>なぜ効くのか（原理）:</b> {why}', s['body_small'])],
            [Paragraph(f'<b>限界・トレードオフ:</b> {tradeoff}', s['body_small'])],
        ]
        tt = Table(td, colWidths=[W])
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), HexColor(L3_COLOR)),
            ('BACKGROUND', (0,1), (0,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([tt, Spacer(1, 1.5*mm)]))

    story.append(Spacer(1, 3*mm))

    # 一時的 vs 永続的の比較表
    story.append(Paragraph('統合方式の比較: 一時的 vs 永続的', s['h2']))
    int_data = [
        [Paragraph('<b>観点</b>', s['th']),
         Paragraph('<b>一時的統合<br/>(プロンプト/コンテキスト)</b>', s['th']),
         Paragraph('<b>永続的統合<br/>(ファインチューニング)</b>', s['th'])],
        [Paragraph('原理', s['tc']),
         Paragraph('in-context learningで推論時に知識を提供', s['tc']),
         Paragraph('勾配降下法でパラメータに知識を焼き込む', s['tc'])],
        [Paragraph('永続性', s['tc']),
         Paragraph('セッション終了で消失', s['tc']),
         Paragraph('モデルに永続的に残る', s['tc'])],
        [Paragraph('更新容易性', s['tc']),
         Paragraph('プロンプト編集で即反映', s['tc']),
         Paragraph('再訓練が必要（時間・コスト大）', s['tc'])],
        [Paragraph('適する知識', s['tc']),
         Paragraph('頻繁に変化する事実・最新情報', s['tc']),
         Paragraph('安定したスタイル・行動パターン', s['tc'])],
        [Paragraph('リスク', s['tc']),
         Paragraph('コンテキストウィンドウの圧迫', s['tc']),
         Paragraph('catastrophic forgetting', s['tc'])],
    ]
    it = Table(int_data, colWidths=[W*0.18, W*0.41, W*0.41])
    it.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 2*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(it)

    story.append(Paragraph(
        '<b>類似性の核心:</b> すべて「モデルに知識を持たせる」が、'
        '保持期間と更新方法が根本的に異なる。多くの実用システムでは'
        'ファインチューニングでスタイルを、プロンプティングで事実を、'
        'マルチモーダルで視覚情報を統合する多層利用が最適解となる。', s['body']))

    story.append(PageBreak())

    # ============ 6. 第4層 — 出力の保証 ============
    story.append(Paragraph('6. 第4層 — 出力の保証: 原理と手法比較', s['h1']))
    story.append(Paragraph(
        'この層は他の3層とは本質的に異なる。他の層が<b>知識をモデルに入れる</b>のに対し、'
        'この層は<b>モデルの出力を検証・制約する</b>。グラウンディングの「最終防衛線」。', s['body']))

    l4_techs = [
        ('ガードレール / Constitutional AI',
         'AIの出力をルールやポリシーに基づきリアルタイムでフィルタリング・修正する。',
         'LLMの出力は確率的であり、常に正確とは限らない。'
         '事後的にルールベースの検証を行うことで、ハルシネーション、有害表現、'
         'ポリシー違反を検出・阻止する。'
         'Constitutional AIは「自分で自分を批判・修正する」メタ認知的アプローチ。',
         '検出はできるが修正は限定的。ルールの網羅性に依存。'
         'レイテンシの増加。正当な出力を誤ってブロックするリスク（偽陽性）。'),
        ('スキーマバリデーション（構造化出力）',
         'AIの出力をJSON Schema等の型定義に適合させる。',
         '出力の「形」を制約することで、下流システムとの統合を保証する。'
         'Structured Outputにより、APIレスポンスとして確実にパース可能な'
         '形式を強制する。',
         '構造の正しさは保証するが、内容の正しさは保証しない。'
         '「正しいJSON形式だが事実と異なる」出力は防げない。'),
        ('形式論理 / ルールエンジン（出力検証として）',
         'AIの出力を述語論理やIF-THENルールで検証する。',
         '形式論理に基づく検証は、矛盾の検出と整合性の保証が可能。'
         '例:「年齢がマイナス」「開始日が終了日より後」等の論理矛盾を'
         '自動検出する。ドメイン固有の業務ルールの強制にも使える。',
         'ルールの定義・維持に専門知識が必要。'
         '定量的な判断（「正確な数値か」）は可能だが、'
         '定性的な判断（「適切な表現か」）は困難。'),
    ]

    for name, what, why, tradeoff in l4_techs:
        td = [
            [Paragraph(f'<b>{name}</b>', ParagraphStyle(
                'TN4', fontName='IPAGothic', fontSize=9.5, leading=14, textColor=white))],
            [Paragraph(f'<b>何をするか:</b> {what}', s['body_small'])],
            [Paragraph(f'<b>なぜ効くのか（原理）:</b> {why}', s['body_small'])],
            [Paragraph(f'<b>限界・トレードオフ:</b> {tradeoff}', s['body_small'])],
        ]
        tt = Table(td, colWidths=[W])
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), HexColor(L4_COLOR)),
            ('BACKGROUND', (0,1), (0,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([tt, Spacer(1, 1.5*mm)]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<b>なぜ「グラウンディング」に含めるのか:</b> '
        'ガードレール等は知識の提供ではないが、「AIの出力を現実に根拠づける」'
        'という広義のグラウンディングの最終段階を担う。'
        '知識を入れるだけでは不十分で、出力が知識と整合しているかの検証が不可欠。'
        'ただし、<b>他の層の手法と同列に比較するのは誤り</b>であり、'
        '補完的な役割として理解すべきである。', s['body']))

    story.append(PageBreak())

    # ============ 7. 層間合成パターン ============
    story.append(Paragraph('7. 層間合成パターン: 手法の組み合わせ方', s['h1']))
    story.append(Paragraph(
        '実用的なグラウンディングは複数の層の手法を合成して構築する。'
        '以下に典型的な合成パターンを示す。', s['body']))

    comp_path = '/tmp/grounding_comp.png'
    generate_composition_chart(comp_path)
    story.append(Image(comp_path, width=W, height=W*0.58))
    story.append(Paragraph('図4: 層間合成パターン — 用途に応じた手法の積み方', s['caption']))

    # パターン詳細
    patterns = [
        ('パターンA: 基本構成（PoC・個人利用）',
         'L2: 平ドキュメント注入 → L3: 構造化プロンプティング',
         'CLAUDE.mdにドメイン知識を書き、プロンプトで推論を誘導する最小構成。'
         'インフラ不要で即日開始可能。小規模では十分だがスケールしない。'),
        ('パターンB: 標準構成（プロダクト）',
         'L1: スキーマ定義 → L2: Advanced RAG → L3: プロンプティング → L4: スキーマ検証',
         'ドキュメントをチャンク化・検索し、構造化出力で応答する標準的なRAGアプリ。'
         'スキーマが入出力の「契約」として機能し、第1層と第4層で品質を挟み込む。'),
        ('パターンC: エンタープライズ構成',
         'L1: KG+オントロジー → L2: Graph RAG+MCP → L3: FT+プロンプト → L4: ガードレール+監査',
         '4層すべてを実装した最も堅牢な構成。金融・医療・法律等の'
         'ミッションクリティカルな領域で必要。導入・維持のコストは最大。'),
        ('パターンD: リアルタイム構成（IoT/Physical AI）',
         'L1: デジタルツイン → L2: MCP+センサー → L3: マルチモーダル → L4: 物理法則検証',
         '物理世界のリアルタイムデータに基づきAIが判断する構成。'
         '製造制御、自動運転、ロボティクス等に適用。'),
    ]

    for ptitle, stack, pdesc in patterns:
        pd = [
            [Paragraph(f'<b>{ptitle}</b>', ParagraphStyle(
                'PN', fontName='IPAGothic', fontSize=10, leading=14, textColor=C_PRIMARY))],
            [Paragraph(f'<b>構成:</b> {stack}', s['body_small'])],
            [Paragraph(pdesc, s['body_small'])],
        ]
        ptab = Table(pd, colWidths=[W])
        ptab.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 2*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ]))
        story.append(KeepTogether([ptab, Spacer(1, 2*mm)]))

    story.append(PageBreak())

    # ============ 8. 技術原理の類似性マトリクス ============
    story.append(Paragraph('8. 技術原理の類似性マトリクス', s['h1']))
    story.append(Paragraph(
        '各手法の原理を5つの観点で整理し、どの手法が「似ている」のかを明確にする。'
        '同じ層内の手法は似ており、異なる層の手法は補完的であることが読み取れる。', s['body']))

    sim_data = [
        [Paragraph('<b>手法</b>', s['th']),
         Paragraph('<b>層</b>', s['th']),
         Paragraph('<b>核心原理</b>', s['th']),
         Paragraph('<b>入力</b>', s['th']),
         Paragraph('<b>出力</b>', s['th']),
         Paragraph('<b>持続性</b>', s['th'])],
    ]
    sim_rows = [
        ('オントロジー',     '1', '論理的公理で概念を定義',   'ドメイン知識', '形式的定義', '永続'),
        ('ナレッジグラフ',   '1', 'トリプルで関係を記述',     'エンティティ', 'グラフ構造', '永続'),
        ('タクソノミー',     '1', '階層分類で概念を整理',     '用語・概念',  '階層木',    '永続'),
        ('スキーマ',        '1', '型と制約でデータの形を定義', 'データ構造',  '型定義',    '永続'),
        ('Naive RAG',      '2', 'ベクトル類似度で検索',     'クエリ',     '関連テキスト', '一時'),
        ('Advanced RAG',   '2', '多段パイプラインで精度向上', 'クエリ',     '精選テキスト', '一時'),
        ('Graph RAG',      '2', 'グラフ走査で関連を辿る',   'クエリ+グラフ','推論チェーン', '一時'),
        ('ツール利用/MCP',  '2', 'API呼び出しでデータ取得',  'ツール定義',  'API応答',    '一時'),
        ('メモリシステム',   '2', '仮想メモリで記憶を管理',   '対話履歴',   '想起された記憶','半永続'),
        ('プロンプティング', '3', 'in-context learningを活用','テンプレート','推論誘導',    '一時'),
        ('ファインチューニング','3','勾配降下法で重みを更新',  '学習データ', 'パラメータ変更', '永続'),
        ('マルチモーダル',   '3', '多感覚統合で理解を補強',   '画像/音声等','統合表現',     '一時'),
        ('ガードレール',    '4', 'ルールベースの事後検証',    'AI出力',    '検証済み出力',  '永続ルール'),
        ('スキーマ検証',    '4', '型適合の強制',           'AI出力',    '構造化出力',   '永続定義'),
        ('ルールエンジン',  '4', '述語論理で矛盾検出',      'AI出力',    '検証結果',     '永続ルール'),
    ]
    layer_colors = {'1': L1_COLOR, '2': L2_COLOR, '3': L3_COLOR, '4': L4_COLOR}
    for row in sim_rows:
        cells = [Paragraph(c, s['tc']) for c in row]
        sim_data.append(cells)

    st = Table(sim_data, colWidths=[W*0.16, W*0.05, W*0.28, W*0.15, W*0.16, W*0.20])
    style_cmds = [
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.4, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 1.2*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 1.5*mm),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ]
    # 層ごとの色帯
    row_idx = 1
    for row in sim_rows:
        lc = HexColor(layer_colors[row[1]])
        style_cmds.append(('BACKGROUND', (1, row_idx), (1, row_idx), lc))
        style_cmds.append(('TEXTCOLOR', (1, row_idx), (1, row_idx), white))
        row_idx += 1
    st.setStyle(TableStyle(style_cmds))
    story.append(st)

    story.append(Spacer(1, 4*mm))

    # 結語
    conc = [[Paragraph(
        '<b>結語:</b> グラウンディング手法の選定は「どの1つを選ぶか」ではなく、'
        '「4つの問い（構造化・配送・統合・保証）にそれぞれどう答えるか」を設計することである。'
        '同じ層内では要件に応じて最適な手法を選択し、異なる層の手法は積み重ねて合成する。'
        'これが本資料の最も重要なメッセージである。',
        ParagraphStyle('Conc', fontName='IPAPGothic', fontSize=10,
                      leading=16, textColor=C_PRIMARY))]]
    ct2 = Table(conc, colWidths=[W])
    ct2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor('#ebf8ff')),
        ('BOX', (0,0), (-1,-1), 1, C_SECONDARY),
        ('TOPPADDING', (0,0), (-1,-1), 4*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 4*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 4*mm),
    ]))
    story.append(ct2)

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF generated: {output_path}")

if __name__ == '__main__':
    build_pdf('/home/user/grounding-research/grounding-techniques-map.pdf')
