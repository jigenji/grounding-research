#!/usr/bin/env python3
"""
L1層 知識の構造化 — 技術原理の深掘りと比較 PDF生成

目的: 図2（表現力×構築コスト）の配置が原理からなぜ導かれるかを説明する
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# フォント
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
FONT_PATH_P = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
pdfmetrics.registerFont(TTFont('IPAGothic', FONT_PATH))
pdfmetrics.registerFont(TTFont('IPAPGothic', FONT_PATH_P))
fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

# カラー
C_PRIMARY    = HexColor('#1a365d')
C_SECONDARY  = HexColor('#2b6cb0')
C_ACCENT     = HexColor('#e53e3e')
C_BG_LIGHT   = HexColor('#f7fafc')
C_BG_HEADER  = HexColor('#2d3748')
C_TEXT       = HexColor('#1a202c')
C_TEXT_LIGHT = HexColor('#4a5568')
C_BORDER     = HexColor('#e2e8f0')
L1_COLOR     = '#2f855a'


def make_styles():
    s = {}
    s['title'] = ParagraphStyle('Title', fontName='IPAGothic', fontSize=24, leading=32,
        textColor=white, alignment=TA_CENTER, spaceAfter=6*mm)
    s['subtitle'] = ParagraphStyle('Subtitle', fontName='IPAPGothic', fontSize=12, leading=18,
        textColor=HexColor('#a0aec0'), alignment=TA_CENTER, spaceAfter=4*mm)
    s['h1'] = ParagraphStyle('H1', fontName='IPAGothic', fontSize=17, leading=24,
        textColor=C_PRIMARY, spaceBefore=6*mm, spaceAfter=4*mm)
    s['h2'] = ParagraphStyle('H2', fontName='IPAGothic', fontSize=13, leading=19,
        textColor=C_SECONDARY, spaceBefore=5*mm, spaceAfter=3*mm)
    s['h3'] = ParagraphStyle('H3', fontName='IPAGothic', fontSize=11, leading=16,
        textColor=C_TEXT, spaceBefore=3*mm, spaceAfter=2*mm)
    s['body'] = ParagraphStyle('Body', fontName='IPAPGothic', fontSize=9.5, leading=16,
        textColor=C_TEXT, spaceAfter=2*mm, alignment=TA_JUSTIFY)
    s['body_s'] = ParagraphStyle('BodyS', fontName='IPAPGothic', fontSize=8.5, leading=14,
        textColor=C_TEXT_LIGHT, spaceAfter=1.5*mm)
    s['bullet'] = ParagraphStyle('Bullet', fontName='IPAPGothic', fontSize=9.5, leading=15,
        textColor=C_TEXT, leftIndent=8*mm, bulletIndent=3*mm, spaceAfter=1*mm)
    s['caption'] = ParagraphStyle('Caption', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT_LIGHT, alignment=TA_CENTER, spaceAfter=2*mm)
    s['toc'] = ParagraphStyle('TOC', fontName='IPAPGothic', fontSize=11, leading=20,
        textColor=C_PRIMARY, leftIndent=5*mm)
    s['th'] = ParagraphStyle('TH', fontName='IPAGothic', fontSize=8.5, leading=12,
        textColor=white, alignment=TA_CENTER)
    s['tc'] = ParagraphStyle('TC', fontName='IPAPGothic', fontSize=8, leading=12, textColor=C_TEXT)
    s['tcc'] = ParagraphStyle('TCC', fontName='IPAPGothic', fontSize=8, leading=12,
        textColor=C_TEXT, alignment=TA_CENTER)
    s['quote'] = ParagraphStyle('Quote', fontName='IPAPGothic', fontSize=9, leading=15,
        textColor=C_TEXT_LIGHT, leftIndent=10*mm, rightIndent=10*mm, spaceAfter=2*mm,
        spaceBefore=2*mm)
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


# ============ 図: 表現力の階段 ============
def generate_expressiveness_ladder(path):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')
    from matplotlib.patches import FancyBboxPatch

    steps = [
        (0.5, 0.5, 1.8, 1.0, '#c6f6d5', '平ドキュメント',
         '構造なし\n(自然言語)', 'O(1)\n検索のみ'),
        (2.3, 1.5, 1.8, 1.0, '#9ae6b4', 'タクソノミー',
         '+階層関係\n(is-a)', 'O(depth)\n木の走査'),
        (4.1, 2.5, 1.8, 1.0, '#68d391', 'スキーマ',
         '+型・制約\n(型検査)', 'O(n)\n検証'),
        (5.9, 3.5, 1.8, 1.0, '#38a169', 'ナレッジグラフ',
         '+任意の関係\n(トリプル)', 'O(V+E)\nグラフ走査'),
        (7.7, 4.5, 1.8, 1.0, '#276749', 'OWLオントロジー',
         '+論理的公理\n(記述論理)', 'PTIME〜\nN2EXPTIME'),
    ]

    for x, y, w, h, color, name, added, complexity in steps:
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', lw=2, alpha=0.85)
        ax.add_patch(r)
        text_c = 'white' if y > 3.0 else '#1a202c'
        ax.text(x+w/2, y+h-0.2, name, fontsize=10, ha='center', va='top',
                color=text_c, fontweight='bold')
        ax.text(x+w/2, y+0.15, added, fontsize=7, ha='center', va='bottom',
                color=text_c, alpha=0.8)
        # 計算量ラベル（右側）
        ax.text(x+w+0.15, y+h/2, complexity, fontsize=7, ha='left', va='center',
                color='#718096')
        # 矢印
        if y > 0.5:
            ax.annotate('', xy=(x+0.2, y+0.1), xytext=(x-0.3, y-0.3),
                        arrowprops=dict(arrowstyle='->', color='#a0aec0', lw=1.5))

    ax.text(5.5, 0.2, '各ステップで追加される表現能力と、それに伴う計算量の増大',
            fontsize=10, ha='center', color='#2d3748', fontweight='bold',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7fafc', edgecolor='#e2e8f0'))

    # 両端ラベル
    ax.text(0.2, 6.3, '表現力: 低\n構築コスト: 低\n計算量: 低', fontsize=8,
            color='#718096', va='top')
    ax.text(9.8, 6.3, '表現力: 高\n構築コスト: 高\n計算量: 高', fontsize=8,
            color='#718096', va='top', ha='right')
    ax.annotate('', xy=(9.5, 5.8), xytext=(0.5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#cbd5e0', lw=2))
    ax.text(5.0, 6.0, '表現力-計算量トレードオフ（Levesque-Brachman, 1987）',
            fontsize=9, ha='center', color='#718096')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


# ============ 図: DL複雑性階層 ============
def generate_dl_complexity(path):
    fig, ax = plt.subplots(figsize=(10, 5))
    from matplotlib.patches import FancyBboxPatch

    levels = [
        (0.5, 'EL++', 'PTIME', 'OWL 2 EL', '#c6f6d5',
         'SNOMED CT\n(35万概念)'),
        (2.3, 'DL-Lite', 'AC0/NLogSpace', 'OWL 2 QL', '#b2f5ea',
         'OBDA\n(SQL還元)'),
        (4.1, 'DLP', 'PTIME(data)', 'OWL 2 RL', '#bee3f8',
         'ルールエンジン\n連携'),
        (5.9, 'ALC', 'EXPTIME', '—', '#c3dafe',
         '基本記述論理'),
        (7.7, 'SROIQ', 'N2EXPTIME', 'OWL 2 DL', '#e9d8fd',
         'フル\nオントロジー'),
        (9.5, 'FOL', '半決定不能', 'OWL 2 Full', '#fed7d7',
         '完全な\n一階述語論理'),
    ]

    for x, dl_name, complexity, owl, color, use in levels:
        r = FancyBboxPatch((x, 0.5), 1.5, 3.5, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', lw=1.5, alpha=0.8)
        ax.add_patch(r)
        ax.text(x+0.75, 3.7, dl_name, fontsize=10, ha='center', va='top',
                fontweight='bold', color='#2d3748')
        ax.text(x+0.75, 2.8, complexity, fontsize=8, ha='center', va='center',
                color='#4a5568')
        ax.text(x+0.75, 2.2, owl, fontsize=7.5, ha='center', va='center',
                color='#718096', fontweight='bold')
        ax.text(x+0.75, 1.2, use, fontsize=7, ha='center', va='center',
                color='#718096')
        if x > 0.5:
            ax.annotate('', xy=(x+0.1, 2.5), xytext=(x-0.3, 2.5),
                        arrowprops=dict(arrowstyle='->', color='#a0aec0', lw=1.2))

    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.text(5.75, 4.3, '記述論理の複雑性階層 — 表現力を追加するたびに計算量クラスが上昇する',
            fontsize=9.5, ha='center', color='#2d3748', fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


# ============ 図: 図2の再導出 ============
def generate_fig2_rederived(path):
    techniques = [
        ("平ドキュメント", 1.0, 1.0, 70, '構造なし\n→検索のみ'),
        ("タクソノミー", 2.0, 1.8, 85, '+階層\n→分類推論'),
        ("スキーマ", 2.5, 2.5, 95, '+型制約\n→検証O(n)'),
        ("ナレッジグラフ", 3.5, 3.5, 110, '+関係グラフ\n→走査O(V+E)'),
        ("OWLオントロジー", 4.8, 4.5, 110, '+論理公理\n→N2EXPTIME'),
        ("形式論理/\nルールエンジン", 4.3, 3.8, 90, '+推論規則\n(直交軸)'),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    # 理論的対角線
    x_line = np.linspace(0.5, 5.5, 100)
    y_line = x_line * 0.95 - 0.05
    ax.plot(x_line, y_line, '--', color='#e2e8f0', lw=2, alpha=0.5)
    ax.text(4.8, 4.0, '理論的対角線\n（表現力-計算量\n　トレードオフ）',
            fontsize=7.5, color='#a0aec0', ha='center', rotation=40)

    # 宣言的知識の進化矢印
    declarative = [0, 1, 2, 3, 4]
    dec_coords = [(techniques[i][1], techniques[i][2]) for i in declarative]
    for i in range(len(dec_coords)-1):
        ax.annotate('', xy=dec_coords[i+1], xytext=dec_coords[i],
                    arrowprops=dict(arrowstyle='->', color='#a0aec0',
                                   lw=1.5, alpha=0.5,
                                   connectionstyle='arc3,rad=0.1'))

    for name, x, y, size, label in techniques:
        is_rule = 'ルール' in name
        color = '#6b46c1' if is_rule else L1_COLOR
        marker = 'D' if is_rule else 'o'
        ax.scatter(x, y, c=color, s=size*1.5, zorder=5, marker=marker,
                   edgecolors='white', linewidth=1.5, alpha=0.9)
        ox, oy = 0.15, 0.18
        if is_rule:
            ox, oy = 0.2, -0.3
        ax.annotate(name, (x, y), fontsize=9, color='#2d3748',
                    xytext=(x+ox, y+oy), fontweight='bold')
        # 理由ラベル
        ax.text(x-0.3, y-0.35, label, fontsize=6.5, color='#718096',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='#e2e8f0', alpha=0.8))

    # 凡例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=L1_COLOR,
               markersize=8, label='宣言的知識（表現力の階段）'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#6b46c1',
               markersize=8, label='推論的知識（直交軸）'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=8,
              framealpha=0.9, edgecolor='#e2e8f0')

    ax.set_xlim(0.3, 5.5)
    ax.set_ylim(0.3, 5.3)
    ax.set_xlabel('知識の表現力（概念・関係・制約の記述能力）  →', fontsize=10, labelpad=8)
    ax.set_ylabel('構築・維持の労力  →', fontsize=10, labelpad=8)
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


# ============ PDF本体 ============
def build_pdf(output_path):
    s = make_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=22*mm,
        title='L1層 知識の構造化 技術原理の深掘りと比較')
    story = []
    W = A4[0] - 40*mm

    # ===== 表紙 =====
    story.append(Spacer(1, 45*mm))
    td = [
        [Paragraph('L1層 知識の構造化', s['title'])],
        [Paragraph('技術原理の深掘りと比較', s['title'])],
        [Spacer(1, 6*mm)],
        [Paragraph('なぜ「表現力×構築コスト」の対角線が生まれるのか<br/>'
                   '— 知識表現仮説と計算量理論からの説明', s['subtitle'])],
    ]
    tt = Table(td, colWidths=[W])
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

    # ===== 目次 =====
    story.append(Paragraph('目次', s['h1']))
    for item in [
        '1. L1層の問い: 知識をどう構造化するか',
        '2. 対角線の理論的根拠: 表現力-計算量トレードオフ',
        '3. 表現力の階段: 各手法の原理深掘り',
        '4. 記述論理の複雑性階層: なぜOWLにプロファイルがあるか',
        '5. 取りこぼし調査: なぜ6手法で十分か',
        '6. 図2の再導出: 原理から配置を説明する',
        '7. 比較マトリクスと選定指針',
    ]:
        story.append(Paragraph(item, s['toc']))
    story.append(PageBreak())

    # ===== 1. L1層の問い =====
    story.append(Paragraph('1. L1層の問い: 知識をどう構造化するか', s['h1']))
    story.append(Paragraph(
        'L1層は「現実世界の知識を、AIが利用できる構造に変換する」ための手法群である。'
        'この層の手法はすべて同じ問い — <b>知識をどう表現するか</b> — に答える。'
        '比較が意味を持つのは、この共通の問いに対する異なるアプローチだからだ。', s['body']))
    story.append(Paragraph(
        '先行資料の図2では、5手法が「表現力×構築コスト」の対角線上に並んだ。'
        'この配置は偶然ではない。計算量理論に基づく必然的帰結である。'
        '本資料では、この対角線がなぜ生じるのかを原理から説明する。', s['body']))

    story.append(Paragraph('中心仮説', s['h2']))
    hyp = [[Paragraph(
        '<b>表現力-計算量トレードオフ:</b> 知識表現言語の表現力が高いほど、'
        'その言語で記述された知識に基づく推論の計算量は必然的に増大する。'
        'これは工学的制約ではなく、数学的必然である。'
        '（Levesque &amp; Brachman, 1987）',
        ParagraphStyle('Hyp', fontName='IPAPGothic', fontSize=10,
                      leading=16, textColor=C_PRIMARY))]]
    ht = Table(hyp, colWidths=[W])
    ht.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor('#ebf8ff')),
        ('BOX', (0,0), (-1,-1), 1, C_SECONDARY),
        ('TOPPADDING', (0,0), (-1,-1), 4*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 4*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 4*mm),
    ]))
    story.append(ht)
    story.append(PageBreak())

    # ===== 2. 理論的根拠 =====
    story.append(Paragraph('2. 対角線の理論的根拠: 表現力-計算量トレードオフ', s['h1']))

    story.append(Paragraph('知識表現仮説（Smith, 1985）', s['h2']))
    story.append(Paragraph(
        'Brian C. Smithが1985年に定式化した仮説。要旨:', s['body']))
    story.append(Paragraph(
        '「機械的に実現された知的プロセスは、(a) 外部観察者がそのプロセスが示す知識の'
        '命題的説明を表現していると自然に解釈する構造的要素を含み、かつ (b) そのような'
        '外部の意味的帰属とは独立に、知識を示す振る舞いを生み出す上で形式的だが因果的かつ'
        '本質的な役割を果たす」', s['quote']))
    story.append(Paragraph(
        'この仮説は2つのことを同時に要求する: 知識を表す構造は (a) 意味的に解釈可能でありながら、'
        '(b) 計算的に活性でなければならない。<b>表現力（a）と計算効率（b）の両立</b>が'
        '知識表現の根本的課題となる。', s['body']))

    story.append(Paragraph('表現力-計算可能性トレードオフ（Levesque &amp; Brachman, 1987）', s['h2']))
    story.append(Paragraph(
        'Levesqueとbrachmanは、このトレードオフが数学的必然であることを証明した。'
        'メカニズムは以下の通り:', s['body']))

    mech_data = [
        [Paragraph('<b>段階</b>', s['th']),
         Paragraph('<b>状況</b>', s['th']),
         Paragraph('<b>推論の計算量</b>', s['th'])],
        [Paragraph('閉世界データベース', s['tc']),
         Paragraph('完全な知識 — 1つのモデルのみ。\n全ての原子命題が真か偽か確定。', s['tc']),
         Paragraph('O(1) 単純な検索', s['tcc'])],
        [Paragraph('不完全知識の許容', s['tc']),
         Paragraph('複数のモデルが可能に。\n「AかBかわからない」を表現可能。', s['tc']),
         Paragraph('全モデルの検査が必要\n→ NP以上', s['tcc'])],
        [Paragraph('+ 選言（OR）', s['tc']),
         Paragraph('「AまたはB」で分岐が発生。\nモデル空間が指数的に爆発。', s['tc']),
         Paragraph('NP困難\n（SAT問題に帰着）', s['tcc'])],
        [Paragraph('+ 否定', s['tc']),
         Paragraph('選言+否定で完全なブール表現力。\n制約の組み合わせ爆発。', s['tc']),
         Paragraph('NP完全以上', s['tcc'])],
        [Paragraph('+ 存在量化', s['tc']),
         Paragraph('「ある X が存在する」で無限ドメインの\n考慮が必要に。', s['tc']),
         Paragraph('PSPACE以上', s['tcc'])],
        [Paragraph('完全な一階述語論理', s['tc']),
         Paragraph('全称・存在量化+全結合子。\n停止しない探索が発生し得る。', s['tc']),
         Paragraph('半決定不能\n（Church-Turing）', s['tcc'])],
    ]
    mt = Table(mech_data, colWidths=[W*0.22, W*0.48, W*0.30])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 2*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(mt)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<b>核心:</b> 表現力の追加子（選言、否定、量化子）を1つ加えるたびに、'
        '推論器が探索すべきモデル空間が組み合わせ的に爆発する。'
        'これは原理的にどんなアルゴリズムでも回避不可能である。', s['body']))

    story.append(Paragraph('記号接地問題との関連（Harnad, 1990）', s['h2']))
    story.append(Paragraph(
        'Harnadの記号接地問題は「記号がどうやって意味を持つか」を問う。'
        'LLMは分布的接地（単語の共起パターン）を持つが、これは「中国語-中国語辞書」と同じで循環的。'
        '知識の構造化は外部接地の足場を提供する:', s['body']))

    ground = [
        ('平ドキュメント', '分布的接地のみ（統計的共起パターン）'),
        ('タクソノミー', '+ 分類的接地（概念の位置づけが固定される）'),
        ('スキーマ', '+ 構造的-機能的接地（データの役割が定義される）'),
        ('ナレッジグラフ', '+ 関係的接地（エンティティの文脈全体が定義される）'),
        ('OWLオントロジー', '+ 推論的接地（論理的に何が導けるかまで定義される）'),
    ]
    for name, desc in ground:
        story.append(Paragraph(f'• <b>{name}:</b> {desc}', s['bullet']))

    story.append(PageBreak())

    # ===== 3. 表現力の階段 =====
    story.append(Paragraph('3. 表現力の階段: 各手法の原理深掘り', s['h1']))
    story.append(Paragraph(
        '各手法を「何が追加され、何が可能になり、なぜコストが増えるか」の3点で深掘りする。', s['body']))

    ladder_path = '/tmp/l1_ladder.png'
    generate_expressiveness_ladder(ladder_path)
    story.append(Image(ladder_path, width=W, height=W*0.54))
    story.append(Paragraph('図1: 表現力の階段 — 各ステップで追加される能力と計算量', s['caption']))

    techs = [
        ('平ドキュメント（Markdown/PDF等）',
         '非構造化テキストをそのまま知識源とする。',
         '<b>数学的基盤:</b> なし（自然言語は形式言語ではない）。'
         'LLMの分布意味論（distributional semantics）に依存し、'
         '単語の意味を共起パターンから推定する。',
         '<b>追加される能力:</b> なし（ベースライン）。人間の自然言語理解能力に全面依存。',
         '<b>計算量:</b> O(1)の単純検索。構造がないため「推論」は不可能。'
         'LLMが推論に見える処理をするが、それはLLM側の能力であり表現側の寄与ではない。',
         'CLAUDE.md、社内Wiki、PDF仕様書。コスト最小だが曖昧さが最大。'),
        ('タクソノミー / 統制語彙',
         '概念を階層的「is-a」関係で分類し、用語を統制する。',
         '<b>数学的基盤:</b> 半順序集合（partially ordered set）。'
         '推移律（AがBの子、BがCの子ならAはCの子孫）と反対称律が成立。'
         'SKOSで形式化される（skos:broader, skos:narrower）。',
         '<b>追加される能力:</b> 階層推論。「電気自動車」を検索すると'
         '自動的に「自動車」の結果も返る。同義語解決（car = automobile）。',
         '<b>計算量:</b> O(depth)の木の走査。多重継承（polyhierarchy）でもO(|V|+|E|)。'
         '計算的にはほぼ無コスト。構築コストは用語の合意形成にかかる。',
         'ICD（国際疾病分類）、UNSPSC（製品分類）、LOC件名標目表。'),
        ('スキーマ / データモデル',
         'データの構造・型・制約を形式的に定義する。',
         '<b>数学的基盤:</b> 型理論（type theory）の実用版。'
         'JSON Schemaは再帰的な型定義、SQL DDLは関係代数に基づく。'
         '制約は述語論理の限定的なサブセット（CHECK制約 = 原子命題の連言）。',
         '<b>追加される能力:</b> 構造検証。AIの出力が「正しい形」であることを'
         '機械的に保証できる（Structured Output）。外部キーで参照整合性を強制。',
         '<b>計算量:</b> 型検査はO(n)。制約検証も通常O(n)。'
         'スキーマ自体に推論能力はないが、下流の検証が確実になる。'
         '構築コストはドメイン理解+設計スキル。スキーマ進化（マイグレーション）のコスト。',
         'JSON Schema（API契約）、SQL DDL（DB設計）、GraphQL型定義。'),
        ('ナレッジグラフ',
         'エンティティと関係のグラフ構造で知識を表現する。',
         '<b>数学的基盤:</b> グラフ理論（有向ラベル付きグラフ）。'
         'RDFトリプル（主語-述語-目的語）の集合としても定式化される。'
         'SPARQLはグラフパターンマッチングに基づくクエリ言語。',
         '<b>追加される能力:</b> マルチホップ推論。'
         '「薬X → 治療 → 疾患Y → 原因 → 遺伝子Z」のような'
         '複数の関係を辿る推論チェーンが構築可能。'
         '説明可能性（回答の根拠パスを提示）。',
         '<b>計算量:</b> グラフ走査O(V+E)。サブグラフ同型判定はNP完全だが、'
         '実用的なクエリはインデックスにより高速化される。'
         '構築コストはエンティティ抽出・解決パイプラインに集中（「IBM」と「Big Blue」の同一性判定等）。',
         'Google Knowledge Graph、Wikidata（1億+項目）、企業内KG。'),
        ('OWLオントロジー',
         '記述論理に基づき概念・関係・公理を厳密に定義する。',
         '<b>数学的基盤:</b> 記述論理（Description Logic）。'
         '一階述語論理の決定可能な断片として設計。'
         'OWL 2 DLはSROIQ(D)に基づき、N2EXPTIME完全。'
         'OWL 2 ELはEL++に基づき、PTIMEで推論可能。',
         '<b>追加される能力:</b> 自動推論。(1)矛盾検出（「薬品かつ食品」を検出）、'
         '(2)自動分類（定義から自動的にクラス階層を導出）、'
         '(3)インスタンス分類（個体がどのクラスに属するか自動判定）。',
         '<b>計算量:</b> OWL 2 EL: PTIME、OWL 2 QL: AC0（SQL還元）、'
         'OWL 2 RL: PTIME、OWL 2 DL: N2EXPTIME。'
         '構築コストはOWL/RDF専門家+ドメインエキスパートの協業。'
         'SNOMED CT（35万概念）の構築に10年以上。',
         'SNOMED CT（医療）、FIBO（金融）、Gene Ontology（生物学）。'),
        ('形式論理 / ルールエンジン（追加手法）',
         '述語論理やIF-THENルールで推論規則を宣言的に定義する。',
         '<b>数学的基盤:</b> 一階述語論理、ホーン節論理（Prolog）、'
         'プロダクションルール（Drools、SWRL）。'
         '前方連鎖（データ→結論）と後方連鎖（目標→前提）の2つの推論戦略。',
         '<b>追加される能力:</b> 手続き的推論。オントロジーが「何であるか」（宣言的）を'
         '定義するのに対し、ルールエンジンは「何をすべきか」（推論的）を定義する。'
         '業務ルール（「18歳未満は購入不可」）の強制。',
         '<b>計算量:</b> ホーン節推論はPTIME。一般的なルール体系はPSPACE以上。'
         '構築コストはルールの網羅性・整合性の維持。ルール間の干渉（意図しない相互作用）が課題。',
         '規制遵守、保険引受、臨床意思決定支援、税計算。'
         '<b>注: 宣言的知識の階段とは直交する軸</b>（後述）。'),
    ]

    for name, what, math, ability, cost, example in techs:
        td = [
            [Paragraph(f'<b>{name}</b>', ParagraphStyle(
                'TN', fontName='IPAGothic', fontSize=9.5, leading=14, textColor=white))],
            [Paragraph(f'<b>何をするか:</b> {what}', s['body_s'])],
            [Paragraph(math, s['body_s'])],
            [Paragraph(ability, s['body_s'])],
            [Paragraph(cost, s['body_s'])],
            [Paragraph(f'<b>実例:</b> {example}', s['body_s'])],
        ]
        bg = '#6b46c1' if 'ルールエンジン' in name else L1_COLOR
        tt = Table(td, colWidths=[W])
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), HexColor(bg)),
            ('BACKGROUND', (0,1), (0,-1), C_BG_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 2.5*mm),
            ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
        ]))
        story.append(KeepTogether([tt, Spacer(1, 1.5*mm)]))

    story.append(PageBreak())

    # ===== 4. 記述論理の複雑性階層 =====
    story.append(Paragraph('4. 記述論理の複雑性階層: なぜOWLにプロファイルがあるか', s['h1']))
    story.append(Paragraph(
        'OWL 2は単一の言語ではなく、表現力-計算量トレードオフの異なる点を選んだ3つのプロファイル'
        '（EL, QL, RL）を提供する。各プロファイルは記述論理の異なる断片に対応する。', s['body']))

    dl_path = '/tmp/l1_dl.png'
    generate_dl_complexity(dl_path)
    story.append(Image(dl_path, width=W, height=W*0.45))
    story.append(Paragraph('図2: 記述論理の複雑性階層 — 各DL断片と対応するOWLプロファイル', s['caption']))

    # 表現子の追加と複雑性上昇
    story.append(Paragraph('表現子の追加と複雑性の上昇', s['h2']))
    cons_data = [
        [Paragraph('<b>表現子</b>', s['th']),
         Paragraph('<b>記号</b>', s['th']),
         Paragraph('<b>何を表現できるようになるか</b>', s['th']),
         Paragraph('<b>複雑性への影響</b>', s['th'])],
        [Paragraph('概念交差', s['tc']), Paragraph('C ⊓ D', s['tcc']),
         Paragraph('「AかつB」（例: 薬品 AND 液体）', s['tc']),
         Paragraph('基本（ALCの一部）', s['tc'])],
        [Paragraph('概念和', s['tc']), Paragraph('C ⊔ D', s['tcc']),
         Paragraph('「AまたはB」（選言 → 分岐爆発）', s['tc']),
         Paragraph('NP困難の原因', s['tc'])],
        [Paragraph('概念否定', s['tc']), Paragraph('¬C', s['tcc']),
         Paragraph('「Aでない」（否定 → 完全ブール）', s['tc']),
         Paragraph('PSPACE以上', s['tc'])],
        [Paragraph('推移的役割', s['tc']), Paragraph('Trans(R)', s['tcc']),
         Paragraph('「partOfの推移律」（部品→組立品→製品）', s['tc']),
         Paragraph('S: ALC+推移', s['tc'])],
        [Paragraph('役割階層', s['tc']), Paragraph('R ⊑ S', s['tcc']),
         Paragraph('「hasFatherはhasParentの下位」', s['tc']),
         Paragraph('H: EXPTIME', s['tc'])],
        [Paragraph('逆役割', s['tc']), Paragraph('R⁻', s['tcc']),
         Paragraph('「hasChildの逆はhasParent」', s['tc']),
         Paragraph('I: EXPTIME', s['tc'])],
        [Paragraph('限定数量制約', s['tc']), Paragraph('≥n R.C', s['tcc']),
         Paragraph('「大学は学長を1人以上持つ」', s['tc']),
         Paragraph('Q: EXPTIME', s['tc'])],
        [Paragraph('複合役割公理', s['tc']), Paragraph('R₁∘R₂ ⊑ S', s['tcc']),
         Paragraph('「叔父 = 親の兄弟」の連鎖定義', s['tc']),
         Paragraph('R: N2EXPTIME', s['tc'])],
    ]
    ct = Table(cons_data, colWidths=[W*0.16, W*0.11, W*0.43, W*0.30])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 1.5*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(ct)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<b>要点:</b> 表現子を1つ追加するたびに、推論の最悪計算量クラスが証明可能に上昇する。'
        'OWLプロファイルは「どの表現子を犠牲にするか」を選ぶ設計選択であり、'
        'これが対角線上のどの点を選ぶかに直結する。', s['body']))

    story.append(PageBreak())

    # ===== 5. 取りこぼし調査 =====
    story.append(Paragraph('5. 取りこぼし調査: なぜ6手法で十分か', s['h1']))
    story.append(Paragraph(
        '調査対象とした候補技術と、採否の判断を以下にまとめる。'
        '結論: ほとんどの候補は既存の6手法に包含される。', s['body']))

    survey_data = [
        [Paragraph('<b>候補技術</b>', s['th']),
         Paragraph('<b>採否</b>', s['th']),
         Paragraph('<b>理由</b>', s['th'])],
        [Paragraph('RDF / RDFS', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('シリアライゼーション形式。裸のRDFはKG、RDFS/OWL付きはオントロジーに包含', s['tc'])],
        [Paragraph('SKOS', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('タクソノミーのエンコーディング標準。構造化パラダイムとしてはタクソノミーそのもの', s['tc'])],
        [Paragraph('セマンティック\nネットワーク', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('ナレッジグラフの歴史的呼称。現代では実質同義', s['tc'])],
        [Paragraph('ER図 / ERモデル', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('データモデリングの記法。実装されるとスキーマ（SQL DDL）になる', s['tc'])],
        [Paragraph('フレーム表現\n（Minsky）', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('OWLクラスとJSON Schemaに吸収済み。スロット+継承 = 現代の型システム', s['tc'])],
        [Paragraph('概念グラフ\n（Sowa）', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('理論的に優美だがRDF/OWLとの標準化競争に敗北。ツール生態系がない', s['tc'])],
        [Paragraph('トピックマップ\n（ISO 13250）', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('KGと構造的に等価。2000年代にRDF/OWLに敗北し事実上終了', s['tc'])],
        [Paragraph('データ辞書 /\nメタデータレジストリ', s['tc']),
         Paragraph('補足的', s['tcc']),
         Paragraph('知識の構造化というよりガバナンス/メタデータ。スキーマの補完として言及', s['tc'])],
        [Paragraph('ディメンション\nモデリング', s['tc']),
         Paragraph('不採用', s['tcc']),
         Paragraph('スキーマ/データモデルの特殊化（スタースキーマ）。独立手法ではない', s['tc'])],
        [Paragraph('<b>形式論理 /\nルールエンジン</b>', s['tc']),
         Paragraph('<b>採用</b>', s['tcc']),
         Paragraph('<b>宣言的知識と直交する推論的知識を構造化。独自の機能カテゴリ</b>', s['tc'])],
    ]
    st = Table(survey_data, colWidths=[W*0.20, W*0.10, W*0.70])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 1.5*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 2*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,-1), (-1,-1), HexColor('#f0fff4')),
    ]))
    story.append(st)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<b>不採用の共通パターン:</b> ほとんどの候補は (1) 既存手法のエンコーディング形式（RDF, SKOS）、'
        '(2) 歴史的な別名（セマンティックネットワーク）、'
        '(3) 特殊化（ディメンションモデリング）、'
        'または (4) 標準化競争に敗北した代替案（概念グラフ、トピックマップ）のいずれか。'
        '6手法は表現力スペクトラムの主要な段階をカバーしている。', s['body']))

    story.append(PageBreak())

    # ===== 6. 図2の再導出 =====
    story.append(Paragraph('6. 図2の再導出: 原理から配置を説明する', s['h1']))
    story.append(Paragraph(
        '以下の図は、先行資料の図2（表現力×構築コスト）を理論的根拠とともに再導出したものである。'
        '各手法の位置は「何が追加されるか」から原理的に決定される。', s['body']))

    fig2_path = '/tmp/l1_fig2_rederived.png'
    generate_fig2_rederived(fig2_path)
    story.append(Image(fig2_path, width=W, height=W*0.7))
    story.append(Paragraph(
        '図3: 図2の再導出 — 各手法の位置はその原理（追加される表現能力と計算量）から決定される', s['caption']))

    story.append(Paragraph('なぜ対角線になるのか — 3つの定理からの帰結', s['h2']))

    theorems = [
        ('<b>Church-Turingの定理（1936-37）:</b> '
         '完全な一階述語論理は決定不能。任意の一階文を捕捉できるほど表現力の高い言語では、'
         '完全な推論アルゴリズムが存在しない。これが表現力の絶対的上限を設定する。'),
        ('<b>Levesque-Brachmanのトレードオフ（1987）:</b> '
         '決定可能な範囲内でも、表現力と計算効率の間にはトレードオフがある。'
         '閉世界データベース（完全知識）は多項式時間だが、不完全知識の許容から'
         '表現子の追加のたびに計算量が証明可能に上昇する。'),
        ('<b>記述論理の複雑性階層（1991-2008）:</b> '
         'EL++(PTIME) → ALC(EXPTIME) → SHOIN(EXPTIME) → SROIQ(N2EXPTIME)。'
         '各表現子の追加がどれだけ計算量を上げるかが精密に定量化されている。'),
    ]
    for t in theorems:
        story.append(Paragraph(f'• {t}', s['bullet']))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('形式論理/ルールエンジンの位置: なぜ直交か', s['h2']))
    story.append(Paragraph(
        '形式論理/ルールエンジンは他の5手法と質的に異なる。'
        '平ドキュメント→オントロジーの階段は<b>宣言的知識</b>（「何であるか」）の表現力を段階的に'
        '高めるのに対し、ルールエンジンは<b>推論的知識</b>（「何をすべきか」「何が導けるか」）を'
        '構造化する。', s['body']))
    story.append(Paragraph(
        'したがって図上では、対角線の「上」ではなく「横」（直交方向）に位置する。'
        'ナレッジグラフ+ルールエンジン、オントロジー+ルールエンジンのように'
        '宣言的知識と推論的知識を組み合わせるのが実用的なアプローチとなる。', s['body']))

    story.append(PageBreak())

    # ===== 7. 比較マトリクス =====
    story.append(Paragraph('7. 比較マトリクスと選定指針', s['h1']))
    story.append(Paragraph(
        '6手法の比較を7つの観点で整理する。', s['body']))

    matrix_data = [
        [Paragraph('<b>観点</b>', s['th']),
         Paragraph('<b>平ドキュ\nメント</b>', s['th']),
         Paragraph('<b>タクソ\nノミー</b>', s['th']),
         Paragraph('<b>スキーマ</b>', s['th']),
         Paragraph('<b>ナレッジ\nグラフ</b>', s['th']),
         Paragraph('<b>OWLオント\nロジー</b>', s['th']),
         Paragraph('<b>ルール\nエンジン</b>', s['th'])],
    ]
    rows = [
        ('表現力', '最低', '低', '中', '高', '最高', '高(直交)'),
        ('計算量', 'O(1)', 'O(depth)', 'O(n)', 'O(V+E)', 'N2EXP', 'PSPACE+'),
        ('構築コスト', '最低', '低', '中', '高', '最高', '高'),
        ('維持コスト', '最低', '低〜中', '中', '高', '最高', '高'),
        ('推論能力', 'なし', '分類', '型検証', '関係走査', '自動推論', 'ルール推論'),
        ('LLM連携', 'プロンプト\n注入', '検索補助\n用語統制', '構造化\n出力', 'Graph\nRAG', 'SPARQL\n+推論', '出力検証\nルール適用'),
        ('典型的\n用途', 'PoC\n個人利用', '製品分類\n用語管理', 'API設計\nDB設計', 'Q&A\n推薦', '医療\n金融', '規制遵守\n業務ルール'),
    ]
    for label, *vals in rows:
        matrix_data.append([Paragraph(f'<b>{label}</b>', s['tc'])] +
                          [Paragraph(v, s['tcc']) for v in vals])

    mx = Table(matrix_data, colWidths=[W*0.12] + [W*0.147]*6)
    mx.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.4, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 1.2*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 1.5*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (-1,0), (-1,-1), HexColor('#f5f0ff')),
    ]))
    story.append(mx)

    story.append(Spacer(1, 4*mm))

    # 選定フロー
    story.append(Paragraph('選定指針', s['h2']))
    guide_data = [
        [Paragraph('<b>要件</b>', s['th']),
         Paragraph('<b>推奨手法</b>', s['th']),
         Paragraph('<b>原理的根拠</b>', s['th'])],
        [Paragraph('まず動かしたい', s['tc']),
         Paragraph('平ドキュメント', s['tc']),
         Paragraph('構造化コスト=0。LLMの分布意味論で十分な場合', s['tc'])],
        [Paragraph('用語の統一が必要', s['tc']),
         Paragraph('タクソノミー', s['tc']),
         Paragraph('半順序の追加で分類推論が可能に。コスト微小', s['tc'])],
        [Paragraph('AIの出力を構造化したい', s['tc']),
         Paragraph('スキーマ', s['tc']),
         Paragraph('型理論に基づく検証。Structured Output の前提条件', s['tc'])],
        [Paragraph('複雑な関連性を\n推論したい', s['tc']),
         Paragraph('ナレッジグラフ', s['tc']),
         Paragraph('グラフ走査で多段推論。Graph RAGの基盤', s['tc'])],
        [Paragraph('自動推論・矛盾検出\nが必要', s['tc']),
         Paragraph('OWLオントロジー', s['tc']),
         Paragraph('記述論理に基づく推論器。医療・金融等の厳密性', s['tc'])],
        [Paragraph('業務ルール・規制\n遵守が必要', s['tc']),
         Paragraph('ルールエンジン\n+ 上記いずれか', s['tc']),
         Paragraph('推論的知識は宣言的知識と直交。組合せが最適', s['tc'])],
    ]
    gt = Table(guide_data, colWidths=[W*0.22, W*0.23, W*0.55])
    gt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BG_HEADER),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 2*mm),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(gt)

    story.append(Spacer(1, 5*mm))

    # 結語
    conc = [[Paragraph(
        '<b>結語:</b> 図2の対角線は「表現力を上げるほどコストが増える」という経験則ではなく、'
        '計算量理論に裏打ちされた数学的必然である。'
        '各手法は表現力の階段の異なる段に位置し、それぞれの段で「何が追加され、何が可能になるか」'
        'が明確に定義される。手法選定とは、この階段のどの段を選ぶかの設計判断に他ならない。',
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
    build_pdf('/home/user/grounding-research/l1-knowledge-structuring-deep-dive.pdf')
