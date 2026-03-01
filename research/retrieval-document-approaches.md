# AIグラウンディング技術の包括的マップ -- 検索ベース・文書ベースアプローチ

## 目次

1. [RAG（検索拡張生成）](#1-rag検索拡張生成retrieval-augmented-generation)
2. [プレーンドキュメント / フラットファイル](#2-プレーンドキュメント--フラットファイル)
3. [エンベディングベースのセマンティック検索](#3-エンベディングベースのセマンティック検索)
4. [ファインチューニング / RLHF（暗黙的グラウンディング）](#4-ファインチューニング--rlhf暗黙的グラウンディング)
5. [ツール利用 / 関数呼び出し / MCP](#5-ツール利用--関数呼び出し--mcp)
6. [構造化プロンプティング](#6-構造化プロンプティング)
7. [技術横断比較マトリクス](#7-技術横断比較マトリクス)

---

## 1. RAG（検索拡張生成 / Retrieval-Augmented Generation）

### 1.1 技術原理

RAGはLLMの「ライブメモリ」として機能し、外部知識ベースから事実に基づいたグラウンディングを提供しながら、言語モデルが流暢で文脈的に適切なテキストを生成する仕組みである。

**基本パイプライン（4段階）：**

| 段階 | 処理内容 | 技術詳細 |
|------|----------|----------|
| **1. 取り込み・インデキシング** | 文書をチャンクに分割し、エンベディングを作成し、ベクトルDBに格納 | チャンクサイズのチューニングが重要（100〜256トークン：精度重視 vs 1024+トークン：文脈完全性重視） |
| **2. 検索（Retrieve）** | セマンティック検索またはハイブリッド検索でTop-K候補を取得 | 密ベクトル + BM25キーワード検索の組み合わせ（RRF: Reciprocal Rank Fusion） |
| **3. リランキング** | クロスエンコーダでリランクし、真の関連性でソート | 2段階検索（高速な密検索 → 高精度なリランカー）でベンチマーク大幅向上 |
| **4. 生成（Generate）** | 引用・文脈を含むプロンプトで回答を生成 | ソースリンク付きの回答が理想 |

**ベクトルエンベディング：**
- トランスフォーマーベースのバイエンコーダが密ベクトルエンベディングを生成
- クエリも同じベクトル空間に埋め込み、類似度検索で最関連チャンクを取得
- 「財務収益（financial earnings）」で検索しても「四半期売上（quarterly revenue）」を取得可能（意味的マッチング）
- 2025年後半のリーダー: Voyage AI（voyage-3-large）がMTEBベンチマークで優位
- マルチモーダルエンベディング: テキストと画像を統一空間に配置可能

**チャンキング戦略：**
- 固定長チャンキング：最も単純だが概念を分断するリスク
- セマンティックチャンキング：固定長より最大9%リコール改善
- 適応的チャンキング：コサイン類似度閾値（>= 0.8）、500語上限、マイクロヘッダー付加
- プロポジションチャンキング：文を論理命題単位に分解
- 核心的なトレードオフ：「精密だが断片的」vs「完全だが曖昧」

### 1.2 RAGバリアント

#### Naive RAG（基礎型）
- **仕組み**: 単純な3ステップ（インデキシング → 検索 → 生成）
- **検索**: TF-IDF/BM25またはシンプルなベクトル検索
- **限界**: 精度（precision）と再現率（recall）の両立が困難、不要なチャンクの取得や必要データの欠落
- **適用場面**: 限定的なスコープのチャットボット、予測可能なFAQシステム

#### Advanced RAG（高度型）
- **仕組み**: リランキング、メモリ、フィードバックループ、分岐、改善されたデータ検索等を組み合わせた洗練版
- **改善点**: クエリ書き換え（query rewriting）、リランキング、ハイブリッド検索
- **移行戦略**: 段階的に導入（クエリ書き換え → リランキング → ハイブリッド検索の順）
- **適用場面**: 本番環境のチャットボット、中規模のナレッジベース

#### Modular RAG（モジュラー型）
- **仕組み**: 検索と生成のプロセスを独立モジュールに分割。リトリーバー、プロセッサ、ジェネレーターを個別に交換・調整可能
- **特徴**: 高いカスタマイズ性、新技術との統合が容易
- **適用場面**: 複数部門向けの大規模エンタープライズシステム、多様なデータソースとの統合

#### Graph RAG（グラフ型）
- **仕組み**: テキストチャンクではなく、エンティティ（ノード）とリレーションシップ（エッジ）のグラフ構造でデータをインデキシング
- **Microsoftの実装**:
  - 第1段階: LLMによるエンティティ・関係性・主張の抽出 → ナレッジグラフ構築
  - 第2段階: Leidenアルゴリズムによる階層的コミュニティ検出 → コミュニティ要約の事前生成
  - クエリ応答: Map-Reduce処理（コミュニティ要約から部分回答を並列生成 → 最終回答に統合）
- **検索モード**: グローバル検索、ローカル検索、DRIFT検索、基本検索
- **性能**: 100万トークン規模のデータセットで、従来RAGベースラインに対し回答の包括性・多様性が大幅改善（マルチホップQAの再現率+6.4ポイント）
- **適用場面**: 知識グラフ、引用ネットワーク、複雑な関連データの推論

#### Agentic RAG（エージェント型）
- **仕組み**: 自律的なエージェントが動的な意思決定とワークフロー最適化を実行
- **特徴**:
  - 自律的意思決定: クエリの複雑さに基づいてエージェントが検索戦略を評価・管理
  - 反復的改善: フィードバックループで検索精度と応答関連性を改善
  - ワークフロー最適化: タスクを動的にオーケストレーション
  - マルチエージェントアーキテクチャ: 異なるエージェントが検索・生成の異なるタスクに特化
- **課題**: エージェント間の連携に高度なオーケストレーションが必要、計算資源の要求が大きい
- **適用場面**: リアルタイム分析、複雑なマルチドメインタスク

#### その他のバリアント
- **RAG-Fusion**: 複数のリフォーミュレーション済みクエリの結果をRRFで統合し再現率向上
- **HyDE (Hypothetical Document Embeddings)**: スパースクエリ向け
- **ColBERT型**: レイトインタラクションとクロスエンコーダリランキングで関連性向上
- **LongRAG**: 圧縮されたロングコンテキストチャンクの検索で粒度認識型検索を実現
- **コンテキストエンジニアリング**: 2025年後半の最注目技術、異なるタスク・場面で最も効果的なコンテキストを動的に組み立てる

### 1.3 評価次元

| 次元 | 評価 |
|------|------|
| **意味的明示性 / 形式性** | 中〜高。チャンク化・エンベディングにより意味が構造化されるが、元文書の意味構造に依存。Graph RAGでは知識グラフとして高度に形式化 |
| **運用結合度 / ガバナンス** | 中〜高。外部知識ベースとの結合が必須。エンタープライズでは監査証跡、ACL、出所追跡等のガバナンスがアーキテクチャに組み込まれる。EU AI Act等の規制対応が必要 |
| **スケーラビリティ** | 高。クエリ時に関連データのみ検索するため本質的にスケーラブル。ただしガバナンス不在ではデータ増大に伴い品質低下 |
| **維持コスト** | 中〜高。初期見積もりの2〜3倍が実態（73%のエンタープライズRAGシステムが予算超過）。ガバナンス税が20〜30%の追加コスト。エンジニアリング時間が総実装予算の25〜40% |
| **典型的な使用例** | エンタープライズナレッジベースQ&A、カスタマーサポート、内部文書検索、規制文書検索、科学研究、リアルタイム分析 |

---

## 2. プレーンドキュメント / フラットファイル

### 2.1 技術原理

プレーンドキュメント（Markdown、PDF、テキストファイル、プロンプトテンプレート）は、LLMへのグラウンディングソースとして最も直接的でシンプルな手法である。ファイルの内容がそのままコンテキストウィンドウに挿入され、モデルの応答を事実・指示・スタイルの面で制約する。

**Markdownが優位な理由：**
- 構造の表現力: 見出し、リスト、テーブル等でPDFの構造を表現でき、LLMが関係性を識別しやすい
- トークン効率: JSON、XML、HTMLより軽量。少ない文字数で意味を伝達 → コンテキスト容量の最大化、コスト削減
- LLMの学習データとの親和性: LLMの訓練コーパスにMarkdownが大量に含まれており、構造化された指示の理解が良好

**主要なパターン：**

#### CLAUDE.md（プロジェクトレベルシステムプロンプト）
- Claude Codeのシステムプロンプトの一部として毎回ロードされる
- コンテキストエンジニアリングの観点から簡潔に保つべき
- ベストプラクティス:
  - タスク固有の指示は別のMarkdownファイルに分離し、自己記述的な名前を付ける
  - CLAUDE.mdには普遍的に適用される指示のみを含め、指示数を最小限に
  - LLMはプロンプトの周縁部（最初と最後）の指示に偏向する傾向がある
  - Claude Codeのシステムプロンプト自体に約50個の指示があり、追加指示の余地は限定的

#### SKILL.md（スキルアーキテクチャ）
- Markdownファイルで定義され、/scripts、/references、/assetsにバンドルファイルを配置
- スキル呼び出し時: Markdownファイルをロード → 詳細指示に展開 → 新しいユーザーメッセージとして会話コンテキストに注入 → 実行コンテキスト（許可ツール、モデル選択）を変更

#### 文書変換パイプライン
- MarkItDown: PDF/Wordからマークダウンへの変換（MCP経由でClaude Desktopに統合可能）
- 高忠実度変換にはPandocが推奨
- LLM-MD: AI会話をMarkdown形式のフラットファイルで定義・共有・バージョン管理するDSL

### 2.2 評価次元

| 次元 | 評価 |
|------|------|
| **意味的明示性 / 形式性** | 低〜中。自然言語ベースで形式的なセマンティクスは限定的。Markdownの構造（見出し、リスト等）で部分的な形式化。意味の解釈はLLMの能力に依存 |
| **運用結合度 / ガバナンス** | 低。ファイルシステムに配置するだけで動作。バージョン管理（Git等）との統合は容易。ガバナンスは手動管理が中心 |
| **スケーラビリティ** | 低〜中。コンテキストウィンドウの制約あり。文書数が増えると全てを含めることが不可能に。大規模ではRAGへの移行が必要 |
| **維持コスト** | 低。テキストエディタで編集可能、特殊なインフラ不要。ただし文書の正確性・最新性の維持は人間の責任 |
| **典型的な使用例** | CLAUDE.md（プロジェクト設定）、システムプロンプト、スキル定義、少数文書のQ&A、プロンプトテンプレート、コーディング規約の伝達 |

---

## 3. エンベディングベースのセマンティック検索

### 3.1 技術原理

エンベディングはAIが意味を理解する基盤である。テキスト、画像、音声、コードを密な数値ベクトルに変換し、類似した概念がベクトル空間上でクラスタリングされる「共通言語」を作り出す。

#### 密ベクトル（Dense Embeddings）
- テキストの意味的内容を捉える固定長の連続値ベクトル
- 全次元に値が存在（例: 512次元、1536次元）
- セマンティックマッチングに優れる: 同義語、パラフレーズ、概念的類似性を捉える
- k-NN検索を使用（メモリ・CPU負荷が高い）
- 推奨次元: 512次元が精度と速度の最適なトレードオフ

#### スパースベクトル（Sparse Embeddings）
- 大部分がゼロ、少数の非ゼロ値のみ（キー・バリューペアで格納）
- 正確なマッチングに優れる（検索エンジン、情報検索）
- 転置インデックスで実装可能（BM25と同等の効率性）
- ニューラルスパース検索: 従来のBM25を学習ベースで改善

#### ハイブリッド検索（2025〜2026年のコンセンサス）
- 密ベクトル検索のみでもBM25のみでも不十分
- **BM25 + ベクトル類似度のハイブリッドが単独手法を一貫して上回る**
- 2024年: 密ベクトル検索のみで「十分」とされていた
- 2026年: ハイブリッドアプローチがRAG精度ベンチマークを倍増させるという業界コンセンサス
- Vespa、Weaviateがビルトインサポートを提供

#### ベクトルデータベース
- 非構造化コンテンツを高次元数値ベクトルとして格納し、クエリベクトルに最も近いアイテムを検索
- 距離がセマンティック類似度を反映するため、正確なワーディングではなく意味で検索可能
- 主要プレイヤー: Pinecone、Weaviate、Milvus、Qdrant
- 2023年は実験的 → 2026年にはコアインフラストラクチャに進化

#### 最新トレンド（2025〜2026年）
- **マルチモーダルエンベディング**: CLIP、ImageBindにより同一ベクトル空間でテキスト・画像・音声を統合検索（Shopifyのビジュアル検索、Spotifyの音声推薦）
- **レイトインタラクションモデル**: ColBERTがトークンごとのベクトルを格納（2〜3倍のストレージコストだが複雑な推論タスクで改善）
- **AIエージェントメモリ**: エージェントの永続的長期メモリとしてベクトルDBを活用

### 3.2 評価次元

| 次元 | 評価 |
|------|------|
| **意味的明示性 / 形式性** | 中。意味がベクトル空間に数学的に符号化されるが、解釈不能（ブラックボックス）。人間が直接読み取れないため、意味の透明性は低い |
| **運用結合度 / ガバナンス** | 中〜高。ベクトルDB、エンベディングモデル、インデキシングパイプラインとの緊密な結合。エンベディングモデルの変更はインデックス全体の再構築を要する |
| **スケーラビリティ** | 高。ANN（近似最近傍）アルゴリズムにより大規模データセットでも効率的検索。マルチモーダル対応で適用範囲拡大 |
| **維持コスト** | 中。ベクトルDBのインフラ費用、エンベディングモデルの更新時のインデックス再構築。ハイブリッド検索の導入で複雑性増加 |
| **典型的な使用例** | セマンティック検索エンジン、類似文書検索、レコメンデーション、RAGの検索コンポーネント、マルチモーダル検索、異常検出 |

---

## 4. ファインチューニング / RLHF（暗黙的グラウンディング）

### 4.1 技術原理

ファインチューニングとRLHF/DPOは、訓練データを通じてモデルのパラメータ自体に知識や振る舞いを「焼き込む」形の暗黙的グラウンディングである。

#### ファインチューニング
- 特殊化されたデータセットでモデルのパラメータを調整し、特定タスクのパフォーマンスを改善
- ドメイン固有の用語・パターン・出力スタイルの学習に有効
- **パラフレーズ発見**: 同一事実の10種類のパラフレーズに曝露すると、1回の曝露より精度が著しく向上

#### RLHF（人間フィードバックからの強化学習）
- 人間の選好に合わせてモデルを調整
- 望ましい出力を言語で記述しにくい場合に特に有効
- DPO（直接選好最適化）: 人間の選好データを直接最適化し、RLHFパイプラインを簡素化

#### 訓練データの限界
- **知識カットオフ**: 2025年1月に訓練されたモデルは2026年2月の出来事を知らない
- **新事実の組み込み困難**: Microsoft研究がLLMファインチューニングの新しい事実情報の組み込み困難を示す。RAGの方が事実更新に有効
- **コスト障壁**: 数千GPU/TPU、数百万ドルのコスト、テラバイト規模のデータ

#### RAGとの比較
| 観点 | ファインチューニング | RAG |
|------|---------------------|-----|
| 知識の鮮度 | 訓練時に固定（静的） | クエリ時に取得（動的） |
| ハルシネーション抑制 | 限定的（内部知識のみに依存） | 効果的（検索データに根拠） |
| コスト効率 | 高コスト（再訓練が必要） | 低コスト（データパイプライン更新のみ） |
| マルチドメイン | ドメインごとに別モデルが必要 | 同一モデルでデータソース切替 |
| 行動・スタイル制御 | 優れている | 限定的 |
| ドメイン専門性 | 深い | 検索品質に依存 |

#### ハイブリッドアプローチ（2025年のベストプラクティス）
- ファインチューニングで流暢性・トーンを調整 + RAGで事実のグラウンディング
- ファインチューニングでRAG情報の活用能力自体を向上させることも可能
- 「ファインチューニング = 行動の形成」「RAG = 事実のグラウンディング」「エージェント = マルチステップ推論」という役割分担

### 4.2 評価次元

| 次元 | 評価 |
|------|------|
| **意味的明示性 / 形式性** | 低。知識がモデルの重みに暗黙的に符号化され、明示的に参照・検証不可能。「ブラックボックス」的グラウンディング |
| **運用結合度 / ガバナンス** | 低〜中。訓練後はモデル単体で動作（外部依存なし）。ただし、どの訓練データがどの出力に影響したかの追跡は極めて困難（監査性が低い） |
| **スケーラビリティ** | 低〜中。ドメインごとに別モデルの訓練・維持が必要な場合は非効率的。大規模な訓練インフラが必要 |
| **維持コスト** | 高。定期的な再訓練、GPU/TPUコスト、データセットの管理。知識更新のたびにコストが発生 |
| **典型的な使用例** | ドメイン固有のスタイル・トーン・用語の習得、出力フォーマットの制御、特定タスクの精度最大化、安全性アライメント |

---

## 5. ツール利用 / 関数呼び出し / MCP

### 5.1 技術原理

ツール利用（Tool Use）と関数呼び出し（Function Calling）は、LLMがリアルタイムで外部APIやサービスにアクセスし、現在の情報に基づいて応答をグラウンディングする手法である。

#### 従来の関数呼び出し
- OpenAIの2023年「function-calling」APIやChatGPTプラグインフレームワーク
- ベンダー固有のコネクタが必要（各LLMのツールスキーマ変更のたびに統合を書き直し）

#### Model Context Protocol（MCP）
- **概要**: Anthropicが2024年11月に導入したオープンスタンダード。AIシステムと外部ツール・データソースの統合方法を標準化
- **解決する問題**: 従来の「N x M」データ統合問題を「N + M」に削減。GitHub、Google Drive、Slackがそれぞれ1つのMCPサーバーを構築するだけで、あらゆるAIツールが接続可能
- **アーキテクチャ**:
  - Language Server Protocol（LSP）のメッセージフローを再利用
  - JSON-RPC 2.0 over WebSocketによる永続的・双方向チャネル
  - ホスト（AIモデル）、クライアント（ブリッジ）、サーバー（ツール・データ）の分離
  - ツールはリアルタイムで発見可能、自己文書化、関数呼び出し/プロンプト注入と互換
- **業界採用状況（2025〜2026年）**:
  - 2025年3月: OpenAIが公式採用
  - 2025年12月: AnthropicがMCPをLinux Foundation傘下のAgentic AI Foundation（AAIF）に寄贈（Anthropic、Block、OpenAI共同設立）
  - 数千のMCPサーバーが構築済み、全主要プログラミング言語のSDKが利用可能
  - **事実上の業界標準**として確立
- **コード実行との統合**: LLMがコードを書いてMCPサーバーと効率的にやりとりする能力を活用。ツールをオンデマンドでロード、モデル到達前にデータをフィルタリング

#### グラウンディングとしての役割
- LLMにリアルタイムの外部データソースへの標準化されたアクセスチャネルを提供
- 検証可能な事実に応答を根拠付け、ハルシネーションを大幅に削減
- 例: Bloomberg（金融）では、リアルタイム市場データ、過去のファイリング、速報ニュースを統一クエリスキーマで統合

#### セキュリティ上の懸念
- ツールは任意のコード実行を意味し、適切な注意が必要
- ツール呼び出し前にユーザーの明示的同意が必須
- 2025年4月: プロンプトインジェクション、ツール権限の問題等のセキュリティ課題が報告

### 5.2 評価次元

| 次元 | 評価 |
|------|------|
| **意味的明示性 / 形式性** | 高。APIスキーマ、関数シグネチャ、JSON-RPCプロトコルにより意味が形式的に定義。ツールの入出力が厳密に型付け |
| **運用結合度 / ガバナンス** | 高。外部サービス・APIとのリアルタイム接続が必須。MCPによりガバナンス（アクセス制御、監査証跡）が標準化されつつあるが、セキュリティリスクも存在 |
| **スケーラビリティ** | 高。MCPの「N+M」モデルにより統合の複雑さが線形にスケール。新ツールの追加が容易 |
| **維持コスト** | 中。MCPサーバーの構築・維持、API変更への追従。ただしMCP標準化によりベンダー固有のコネクタ維持コストは削減 |
| **典型的な使用例** | リアルタイムデータ取得（天気、株価、ニュース）、外部サービス操作（メール送信、予約）、コード実行、データベースクエリ、ファイルシステム操作 |

---

## 6. 構造化プロンプティング

### 6.1 技術原理

構造化プロンプティングは、プロンプトの設計・構造を通じてLLMの応答をグラウンディングする手法群である。モデルの重みを更新せずに、出力の正確性・一貫性・形式を制御する。

#### Chain-of-Thought（CoT）プロンプティング
- **原理**: 最終回答に至る前に一連の中間ステップとして問題を解決させる
- **効果**: LLMは知識不足ではなく推論ステップの省略で誤答することが多い。CoTで思考過程を露出させ、出力の正確性・監査可能性・信頼性を向上
- **Zero-shot CoT**: 「Let's think step by step（段階的に考えましょう）」を追加するだけで効果
- **Few-shot CoT**: 手作業のステップバイステップ推論の例を1〜2個提供
- **自動CoT（Zhang et al., 2022）**: 「Let's think step by step」プロンプトでLLM自身にデモンストレーション用の推論チェーンを生成させる

#### Few-shot プロンプティング
- 出力フォーマット、トーン、構造、振る舞いの模倣に有効
- CoTとの組み合わせが特に効果的
- 構造化プロンプティング（Hao et al., 2022）: In-Contextラーニングを1,000例にスケーリング

#### Self-Consistency プロンプティング
- 単一の推論パスに依存せず、複数の推論パスを生成し、最も一貫した回答を選択
- CoT推論の精度を向上

#### Tree-of-Thought
- CoTを一般化し、複数の推論ラインを並列に生成
- バックトラックや他のパスの探索が可能
- 幅優先、深さ優先、ビーム探索等の木探索アルゴリズムを使用

#### Chain of Grounded Objectives（CGO）
- LLMが最終コードを生成する前に、中間的な補足コンテンツ（目的、計画、CoT、疑似コード）を自律的に生成
- 訓練データ中のコメント・ドキュメンテーションの構造化された説明パターンを活用

#### メタプロンプティング
- LLMの応答をより組織化・効率化するための構造化・ガイド手法
- タスク横断的な一般化を支援

#### 構造化出力フォーマット
- JSON、YAML、テーブル等の出力フォーマットを指定することで、応答の構造を制約
- 下流システムとの統合を容易にし、パース可能な出力を保証

### 6.2 評価次元

| 次元 | 評価 |
|------|------|
| **意味的明示性 / 形式性** | 中。プロンプト内の自然言語指示で意味を伝達。構造化出力フォーマットの指定により部分的に形式化。ただし解釈のばらつきは避けられない |
| **運用結合度 / ガバナンス** | 低。プロンプトのみで実装可能、外部システムとの結合不要。ただしプロンプトの品質管理・バージョン管理が課題。効果はモデル依存で経験的 |
| **スケーラビリティ** | 中。プロンプトテンプレートの再利用で一定のスケール。しかしFew-shotの例はコンテキストウィンドウを消費。タスクごとのプロンプトエンジニアリングが必要 |
| **維持コスト** | 低。テキスト編集のみ、インフラ不要。ただしモデルアップデート時にプロンプトの再検証が必要（経験科学的性質） |
| **典型的な使用例** | 複雑な推論タスク（数学、論理）、構造化データ抽出、コード生成、出力フォーマット制御、一貫性のある応答生成、監査可能な意思決定過程の記録 |

---

## 7. 技術横断比較マトリクス

### 7.1 総合比較表

| 技術 | 意味的明示性/形式性 | 運用結合度/ガバナンス | スケーラビリティ | 維持コスト | 知識の鮮度 |
|------|---------------------|----------------------|-----------------|-----------|-----------|
| **RAG（Naive）** | 中 | 中 | 中 | 低〜中 | リアルタイム |
| **RAG（Advanced/Modular）** | 中〜高 | 中〜高 | 高 | 中〜高 | リアルタイム |
| **Graph RAG** | 高 | 高 | 中〜高 | 高 | リアルタイム |
| **Agentic RAG** | 中〜高 | 最高 | 高 | 最高 | リアルタイム |
| **プレーンドキュメント** | 低〜中 | 低 | 低〜中 | 低 | 手動更新 |
| **セマンティック検索** | 中 | 中〜高 | 高 | 中 | インデックス更新時 |
| **ファインチューニング/RLHF** | 低 | 低〜中 | 低〜中 | 高 | 訓練時に固定 |
| **ツール利用/MCP** | 高 | 高 | 高 | 中 | リアルタイム |
| **構造化プロンプティング** | 中 | 低 | 中 | 低 | なし（モデル知識に依存） |

### 7.2 適用場面の選択ガイド

```
要件分析フロー:

[リアルタイムの外部データが必要?]
  ├─ Yes → [構造化APIアクセス?] → Yes → ツール利用/MCP
  │                              → No  → RAG（Advanced/Agentic）
  │
  └─ No → [大量のドメイン文書がある?]
            ├─ Yes → [複雑な関連性推論が必要?]
            │         ├─ Yes → Graph RAG
            │         └─ No  → RAG（Naive/Advanced）+ セマンティック検索
            │
            └─ No → [モデルの行動・スタイル変更が必要?]
                      ├─ Yes → ファインチューニング/RLHF
                      └─ No  → [少数の文書/指示で十分?]
                                ├─ Yes → プレーンドキュメント + 構造化プロンプティング
                                └─ No  → ハイブリッドアプローチ
```

### 7.3 2025〜2026年の主要トレンド

1. **コンテキストエンジニアリングの台頭**: 異なるタスク・場面で最も効果的なコンテキストを動的・知的に組み立てる技術が2025年後半の最注目領域
2. **RAGのエンタープライズ標準化**: RAGは「機能するか」ではなく「安全・検証可能・統治可能にどうスケールするか」の段階へ
3. **MCPの業界標準確立**: AIモデルとツール・データの接続における事実上の標準として、Linux Foundation傘下で発展
4. **ハイブリッド検索の必須化**: 密ベクトル + スパース検索の組み合わせが2026年の業界コンセンサス
5. **ガバナンスのアーキテクチャ組み込み**: 後付けではなく、設計段階からガバナンスを組み込む（EU AI Act等への対応）
6. **マルチモーダル・エンベディング**: テキスト・画像・音声の統一ベクトル空間が実用段階へ
7. **エージェント + RAGの融合**: エージェントが「いつ・どのように検索するか」をオーケストレーションし、RAGがグラウンディングメカニズムとして機能

---

## 参考文献・情報源

### RAG全般
- [Microsoft: Common RAG Techniques Explained](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/04/common-retrieval-augmented-generation-rag-techniques-explained/)
- [Data Nucleus: RAG Enterprise Guide 2025](https://datanucleus.dev/rag-and-agentic-ai/what-is-rag-enterprise-guide-2025)
- [RAGFlow: From RAG to Context - 2025 Review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [Pinecone: Retrieval-Augmented Generation](https://www.pinecone.io/learn/retrieval-augmented-generation/)

### RAGバリアント比較
- [DigitalOcean: RAG, AI Agents, and Agentic RAG Comparative Analysis](https://www.digitalocean.com/community/conceptual-articles/rag-ai-agents-agentic-rag-comparative-analysis)
- [Meilisearch: Naive RAG vs Advanced RAG](https://www.meilisearch.com/blog/naive-rag-vs-advanced-rag)
- [Meilisearch: 14 Types of RAG](https://www.meilisearch.com/blog/rag-types)
- [NVIDIA: Traditional RAG vs Agentic RAG](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/)
- [arXiv: Agentic RAG Survey](https://arxiv.org/html/2501.09136v1)

### Graph RAG
- [Microsoft Research: GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/)
- [arXiv: From Local to Global - GraphRAG](https://arxiv.org/abs/2404.16130)
- [Neo4j: GraphRAG Field Guide](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)

### プレーンドキュメント / CLAUDE.md
- [Anthropic: Using CLAUDE.MD Files](https://claude.com/blog/using-claude-md-files)
- [HumanLayer: Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Claude Agent Skills Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)
- [Wetrocloud: Why Markdown is the Best Format for LLMs](https://medium.com/@wetrocloud/why-markdown-is-the-best-format-for-llms-aa0514a409a7)

### エンベディング / セマンティック検索
- [ArtSmart: Top Embedding Models in 2025](https://artsmart.ai/blog/top-embedding-models-in-2025/)
- [Elastic: Sparse Vector Embedding](https://www.elastic.co/search-labs/blog/sparse-vector-embedding)
- [Zilliz: Sparse and Dense Embeddings](https://zilliz.com/learn/sparse-and-dense-embeddings)
- [Cubitrek: Hybrid Search Optimization](https://cubitrek.com/blog/hybrid-search-optimization-how-bm25-and-dense-vector-retrieval-work-together-for-superior-ai-search/)
- [Meilisearch: What Are Vector Embeddings](https://www.meilisearch.com/blog/what-are-vector-embeddings)

### ファインチューニング / RLHF
- [Monte Carlo Data: RAG vs Fine Tuning](https://www.montecarlodata.com/blog-rag-vs-fine-tuning/)
- [Google Cloud: To Tune or Not to Tune](https://cloud.google.com/blog/products/ai-machine-learning/to-tune-or-not-to-tune-a-guide-to-leveraging-your-data-with-llms)
- [MITRIX: Fine-tuning vs RAG vs Agents](https://mitrix.io/blog/llm-fine%E2%80%91tuning-vs-rag-vs-agents-a-practical-comparison/)
- [Superteams: Fine-Tuning and Grounding of LLMs](https://www.superteams.ai/blog/fine-tuning-and-grounding-of-large-language-models-for-enhanced-performance)

### ツール利用 / MCP
- [Wikipedia: Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol)
- [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25)
- [SerpAPI: MCP Unified Standard](https://serpapi.com/blog/model-context-protocol-mcp-a-unified-standard-for-ai-agents-and-tools/)
- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [Bloomberg: MCP in Financial Services](https://glama.ai/blog/2025-11-06-engineering-enterprise-grade-context-making-the-model-context-protocol-mcp-viable-for-financial-services)

### 構造化プロンプティング
- [Prompting Guide: Chain-of-Thought](https://www.promptingguide.ai/techniques/cot)
- [K2view: Prompt Engineering Techniques for 2026](https://www.k2view.com/blog/prompt-engineering-techniques/)
- [Lakera: Ultimate Guide to Prompt Engineering 2026](https://www.lakera.ai/blog/prompt-engineering-guide)
- [Patronus: Advanced Prompt Engineering Techniques](https://www.patronus.ai/llm-testing/advanced-prompt-engineering-techniques)

### ガバナンス / エンタープライズ
- [Enterprise Knowledge: Data Governance for RAG](https://enterprise-knowledge.com/data-governance-for-retrieval-augmented-generation-rag/)
- [APXML: Data Governance and Lineage in RAG Systems](https://apxml.com/courses/optimizing-rag-for-production/chapter-7-rag-scalability-reliability-maintainability/data-governance-lineage-rag)
- [Amit Kothari: The Hidden Costs of RAG](https://amitkoth.com/hidden-costs-rag/)
- [Techment: RAG in 2026 for Enterprise AI](https://www.techment.com/blogs/rag-models-2026-enterprise-ai/)

### AIグラウンディング全般
- [ODSC: What is Grounding in AI](https://odsc.medium.com/what-is-grounding-in-ai-and-what-are-the-best-techniques-655e985cc06f)
- [Microsoft: Grounding Data Design for AI Workloads](https://learn.microsoft.com/en-us/azure/well-architected/ai/grounding-data-design)
- [arXiv: Grounding for Artificial Intelligence](https://arxiv.org/html/2312.09532v1)
