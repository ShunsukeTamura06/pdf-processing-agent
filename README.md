# PDF処理パイプライン with エージェント

LangChainとLangGraphを使用したPDF処理パイプラインとエージェントベースのツール実行の実装です。

## 概要

このプロジェクトは、以下の機能を提供します：

1. PDFファイルからテキストを抽出
2. 抽出したテキストを日本語に翻訳 
3. 翻訳したテキストを要約
4. エージェント判断による条件分岐と適切なツール実行
   - 住所検出 → 天気情報取得
   - メールアドレス検出 → メール送信

## フロー図

```
graph TD
    A[find_pdfs<br>PDFファイル検索] --> B{should_continue?<br>処理継続判定};
    B -- Yes (ファイルあり) --> C[extract_text<br>テキスト抽出];
    C --> D[translate<br>翻訳];
    D --> E[summarize<br>要約];
    E --> H[agent_decision<br>エージェント判断<br>(住所/メール検出)];
    H --> I{route_after_agent<br>ツール要否判定};
    I -- weatherTool --> J[execute_weather_tool<br>天気情報取得];
    I -- mailTool --> K[execute_mail_tool<br>メール送信];
    I -- no tool --> F[store_result<br>結果保存];
    J --> F;
    K --> F;
    F --> B;
    B -- No (全ファイル完了) --> G((END<br>終了));
```

## 主要コンポーネント

- **LangChain**: LLMとの対話、ツール定義、エージェント実行
- **LangGraph**: 状態管理とフロー制御
- **OpenAI GPT-4o**: テキスト処理とエージェント判断
- **PyPDF**: PDFファイルからのテキスト抽出

## 使用しているツール

1. **weatherTool**: 指定された住所の天気情報を取得
2. **mailTool**: 指定されたメールアドレスにメールを送信

## 必要なライブラリ

```bash
pip install pypdf langchain langchain_openai langgraph python-dotenv reportlab duckduckgo-search
```

## 使用方法

1. 環境変数に`OPENAI_API_KEY`を設定
2. コードを実行すると、サンプルPDFが自動生成され、パイプラインが実行されます

```bash
python main.py
```

## 拡張性

このプロジェクトは以下のような拡張が可能です：

- 新しいツールの追加（例：地図表示、翻訳など）
- 異なるLLMモデルの使用
- 並列処理による効率化
- UIの追加

## 注意点

- 実際の利用には、適切なAPI設定とエラーハンドリングが必要です
- PDFのフォント設定は環境に依存する場合があります
