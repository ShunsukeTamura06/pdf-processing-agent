import os
import glob
import re
from typing import List, TypedDict, Annotated, Sequence, Optional
import operator

# PDF抽出ライブラリ (例: pypdf)
# pip install pypdf langchain langchain_openai langgraph python-dotenv reportlab duckduckgo-search # duckduckgoはツール例用
from pypdf import PdfReader
# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool # Tool定義用
from langchain import hub # Agentプロンプト取得用
from langchain.agents import AgentExecutor, create_openai_tools_agent # Agent生成用

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # 結果をメモリに保存する場合
from langgraph.prebuilt import ToolNode # ToolNodeを使うとツール実行が簡単

# --- 状態定義 (拡張) ---
class GraphState(TypedDict):
    """
    グラフ全体で共有される状態 (エージェント/ツール対応版)。
    """
    folder_path: str
    pdf_files: List[str]
    current_index: int
    current_file: str | None
    extracted_text: str | None
    translated_text: str | None
    summary: str | None
    # --- エージェント/ツール関連 ---
    agent_outcome: Optional[dict] # エージェントの最終出力 (ToolMessageなどを含む可能性)
    tool_output: Optional[str] # 実行されたツールの出力テキスト
    # --- 結果/エラー ---
    results: List[dict]
    error: str | None

# --- ツールの定義 (例) ---
# 実際のツールは外部API連携などが必要になります

@tool
def weatherTool(address: str) -> str:
    """
    指定された住所の現在の天気を取得します。
    例: "東京都千代田区", "大阪市北区"
    """
    print(f"--- weatherTool実行 (住所: {address}) ---")
    # ここで実際の天気APIを呼び出す (例: OpenWeatherMap API)
    # ダミー実装:
    if "東京" in address:
        return f"{address}の天気は晴れです。"
    elif "大阪" in address:
        return f"{address}の天気は曇りです。"
    else:
        # DuckDuckGoで検索してみる例 (より汎用的に)
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            weather_info = search.run(f"{address}の天気")
            return weather_info
        except ImportError:
            return f"{address}の天気情報を取得できませんでした（DuckDuckGoツール未インストール）。"
        except Exception as e:
             return f"{address}の天気情報を取得中にエラーが発生しました: {e}"

@tool
def mailTool(email_address: str, subject: str, body: str) -> str:
    """
    指定されたメールアドレスに、指定された件名と本文でメールを送信します。
    """
    print(f"--- mailTool実行 (宛先: {email_address}) ---")
    print(f"  件名: {subject}")
    print(f"  本文: {body[:100]}...") # 長い本文は省略表示
    # ここで実際のメール送信APIやSMTPライブラリを呼び出す
    # ダミー実装:
    # 簡単なメールアドレス形式チェック
    if re.match(r"[^@]+@[^@]+\.[^@]+", email_address):
        # 実際の送信処理は行わず、成功したとみなす
        return f"{email_address} へのメール送信をシミュレートしました。"
    else:
        return f"無効なメールアドレス形式です: {email_address}"

# 利用するツールのリスト
tools = [weatherTool, mailTool]

# --- LLMとエージェントの準備 ---
try:
    # OpenAI GPT-4o を利用 (Tools呼び出しに対応)
    # 環境変数 `OPENAI_API_KEY` にAPIキーを設定してください。
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Agentのプロンプトを取得 (OpenAI Tools Agent用)
    # https://smith.langchain.com/hub/hwchase17/openai-tools-agent
    prompt = hub.pull("hwchase17/openai-tools-agent")

    # Agentを生成
    agent_runnable = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True) # verbose=Trueで動作ログ表示

except ImportError:
    print("必要なライブラリが見つかりません。`pip install langchain_openai duckduckgo-search langchainhub` などを実行してください。")
    llm = None
    agent_executor = None
except Exception as e:
    print(f"LLMまたはエージェントの初期化中にエラーが発生しました: {e}")
    llm = None
    agent_executor = None

# --- ノード関数 (既存ノードは省略、変更/追加分のみ) ---

def summarize_text(state: GraphState) -> GraphState:
    """翻訳された日本語テキストを要約するノード"""
    print("--- テキスト要約中 ---")
    text_to_summarize = state['translated_text']
    if not text_to_summarize:
        print("要約するテキストがありません。スキップします。")
        return {**state, "summary": None, "error": "要約対象テキストなし"}

    if not llm:
        return {**state, "error": "LLMが初期化されていません。"}

    messages = [
        SystemMessage(content="あなたは、与えられた日本語テキストを簡潔に要約するアシスタントです。"),
        HumanMessage(content=f"以下の日本語テキストを3～5文程度で要約してください:\n\n{text_to_summarize}")
    ]
    try:
        response = llm.invoke(messages)
        summary = response.content
        print("要約完了")
        # 次のエージェント処理のために状態をクリア
        return {**state, "summary": summary, "agent_outcome": None, "tool_output": None, "error": None}
    except Exception as e:
        print(f"エラー: 要約中にエラーが発生しました: {e}")
        return {**state, "summary": None, "error": f"要約エラー: {e}"}

# --- 新しいノード関数 ---

def agent_decision_node(state: GraphState) -> GraphState:
    """エージェントが要約を分析し、ツールを呼び出すか判断するノード"""
    print("--- エージェント判断中 ---")
    summary = state['summary']
    if not summary:
        print("要約がないため、エージェント判断をスキップします。")
        # agent_outcome を None または特定の形式にしておく
        return {**state, "agent_outcome": {"output": "要約がないためスキップ"}}

    if not agent_executor:
         return {**state, "error": "エージェントが初期化されていません。"}

    # エージェントへの入力を作成
    # "input" キーがデフォルトで使われることが多い
    agent_input = {
        "input": f"以下の要約テキストを分析してください。\nもし住所（日本の地名など）が含まれていれば、weatherToolを使ってその場所の天気を調べてください。\nもしメールアドレスが含まれていれば、mailToolを使って、件名「要約情報」本文「{summary[:50]}...」でメールを送信してください。\nどちらもなければ、何もせず終了してください。\n\n要約:\n{summary}"
    }

    try:
        # エージェントを実行
        # AgentExecutor は辞書型の出力を返す (通常 'output' キーに最終結果)
        agent_outcome = agent_executor.invoke(agent_input)
        print(f"エージェント実行結果: {agent_outcome}")
        return {**state, "agent_outcome": agent_outcome, "error": None} # agent_outcome にツール呼び出し情報が含まれる
    except Exception as e:
        print(f"エラー: エージェント実行中にエラーが発生しました: {e}")
        return {**state, "error": f"エージェント実行エラー: {e}"}

# ToolNode: エージェントが要求したツールを実行する
# agent_outcome からツール呼び出し情報を読み取り、対応するツールを実行し、
# 結果を ToolMessage として agent_outcome に追加（または上書き）する想定
# LangGraph v0.2以降では、AgentExecutorの出力をそのままToolNodeに渡せる場合がある
tool_node = ToolNode(tools)

def store_result_and_increment(state: GraphState) -> GraphState:
    """処理結果（ツール出力含む）を保存し、次のファイルのインデックスに進めるノード"""
    print("--- 結果を保存 ---")

    # agent_outcomeからツールの最終出力を取得しようと試みる
    tool_output_text = None
    agent_outcome = state.get("agent_outcome")
    if isinstance(agent_outcome, dict) and "output" in agent_outcome:
         # AgentExecutorのinvokeの直接の結果の場合
         tool_output_text = agent_outcome.get("output")
         # ToolNodeを経由した場合、ToolMessageを探す必要があるかもしれない
         # このあたりはAgentExecutorやToolNodeの具体的な動作に依存
         # ここでは簡略化のため、agent_outcome["output"] をツール結果とする

    current_result = {
        "file": state['current_file'],
        "summary": state['summary'],
        "tool_output": tool_output_text, # ツールが実行された場合の出力
        "error": state['error'] # このファイル処理中のエラー
    }
    updated_results = state['results'] + [current_result]
    next_index = state['current_index'] + 1

    # 状態をリセットして次のループへ
    return {
        **state,
        "results": updated_results,
        "current_index": next_index,
        "agent_outcome": None, # 次のファイルのためにクリア
        "tool_output": None,   # 次のファイルのためにクリア
        # エラーは個々の結果に含めた
    }

# --- 条件分岐関数 (追加/変更) ---

def route_after_agent(state: GraphState) -> str:
    """エージェントの判断結果に基づき、ツール実行か結果保存かを決定"""
    print("--- エージェント後の経路判定 ---")
    agent_outcome = state.get("agent_outcome")

    # agent_decision_nodeがエラーを返した場合などは直接storeへ
    if state.get("error"):
         print("エラー発生のため結果保存へ")
         return "store_result"

    # LangGraph v0.2以降のAgentExecutor + ToolNodeの一般的な使い方:
    # agent_outcomeにToolMessageが含まれていればToolNodeへ、そうでなければ終了(結果保存)へ
    if isinstance(agent_outcome, dict) and agent_outcome.get("intermediate_steps"):
         # OpenAI Tools Agent などでは intermediate_steps にツール呼び出しが入る
         print("ツール呼び出しが必要と判断 -> tool_nodeへ")
         return "tool_node" # ToolNodeのデフォルト名
    elif isinstance(agent_outcome, dict) and "tool_calls" in agent_outcome.get("log", ""): # ReAct Agentなど
         print("ツール呼び出しが必要と判断 -> tool_nodeへ")
         return "tool_node"
    else:
         # ツール呼び出しが不要、またはエージェントが直接回答した場合
         print("ツール呼び出し不要と判断 -> store_resultへ")
         return "store_result"


def should_continue(state: GraphState) -> str:
    """処理を続けるか、終了するかを決定する (変更なし)"""
    print("--- 継続/終了判定 ---")
    if state['error'] and state['current_file'] is None:
        print(f"初期エラーのため終了: {state['error']}")
        return "end_processing"

    index = state['current_index']
    pdf_files = state['pdf_files']

    if index < len(pdf_files):
        print("次のファイルの処理へ")
        return "continue_processing"
    else:
        print("全ファイル処理完了のため終了")
        return "end_processing"

# --- グラフ構築 (更新) ---
workflow = StateGraph(GraphState)

# 既存ノードの追加 (関数名は前のコードと同じとする)
# (find_pdf_files, select_file_to_process, extract_text_from_pdf, translate_text は前のコードから流用)
# workflow.add_node("find_pdfs", find_pdf_files)
# workflow.add_node("select_file", select_file_to_process)
# workflow.add_node("extract_text", extract_text_from_pdf)
# workflow.add_node("translate", translate_text)
workflow.add_node("summarize", summarize_text) # summarizeは更新

# 新しいノードの追加
workflow.add_node("agent_node", agent_decision_node)
workflow.add_node("tool_node", tool_node) # ToolNodeを追加
workflow.add_node("store_result", store_result_and_increment) # store_resultは更新

# --- ここからは前のコードのノード定義関数が必要 ---
# (簡単のため、ここではコメントアウト。実際には前のコードの関数定義が必要)
import tempfile
def find_pdf_files(state: GraphState) -> GraphState:
    print("--- PDFファイル検索中 ---")
    folder_path = state['folder_path']
    if not os.path.isdir(folder_path): return {**state, "error": f"エラー: パス '{folder_path}' はフォルダではありません。"}
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    print(f"{len(pdf_files)} 個のPDFファイルが見つかりました。")
    if not pdf_files:
        # サンプルファイル作成ロジック (前のコードから)
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            def create_dummy_pdf(filename, content):
                c = canvas.Canvas(filename, pagesize=letter)
                # 日本語対応のためフォント設定 (環境に依存)
                try:
                    from reportlab.pdfbase import pdfmetrics
                    from reportlab.pdfbase.ttfonts import TTFont
                    # 例: WindowsのMSゴシック (パスは環境に合わせて変更)
                    # font_path = "C:/Windows/Fonts/msgothic.ttc"
                    # if os.path.exists(font_path):
                    #     pdfmetrics.registerFont(TTFont('MS-Gothic', font_path))
                    #     c.setFont('MS-Gothic', 12)
                    # else:
                    #     print("警告: MSゴシックフォントが見つかりません。日本語が文字化けする可能性があります。")
                    #     c.setFont("Helvetica", 12) # デフォルトフォント
                    # 簡単のため英語で作成
                    c.setFont("Helvetica", 12)

                except ImportError:
                     print("警告: reportlab.pdfbase が見つかりません。日本語フォント設定をスキップします。")
                     c.setFont("Helvetica", 12)

                # サンプル内容に住所やメールアドレスを入れてみる
                text_lines = content.split('\n')
                y_position = 750
                for line in text_lines:
                    c.drawString(100, y_position, line)
                    y_position -= 15 # Adjust line spacing if needed
                c.save()

            dummy_content1 = "This document discusses the weather in Tokyo.\nPlease check the weather at 東京都千代田区."
            dummy_content2 = "Contact information: user@example.com.\nPlease send a confirmation email."
            dummy_content3 = "Report about Osaka office.\nLocation: 大阪市北区 1-1-1" # 天気ツール用
            dummy_content4 = "General discussion document." # ツール不要

            create_dummy_pdf(os.path.join(folder_path, "dummy_weather_tokyo.pdf"), dummy_content1)
            create_dummy_pdf(os.path.join(folder_path, "dummy_email.pdf"), dummy_content2)
            create_dummy_pdf(os.path.join(folder_path, "dummy_weather_osaka.pdf"), dummy_content3)
            create_dummy_pdf(os.path.join(folder_path, "dummy_no_tool.pdf"), dummy_content4)
            print("サンプルPDFファイルを4つ作成しました。")
            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf")) # 再検索
        except ImportError:
            print("reportlab がインストールされていません。サンプルPDFは作成されません。")
            return {**state, "error": f"エラー: フォルダ '{folder_path}' にPDFファイルが見つかりません。"}
        except Exception as e:
            print(f"サンプルPDF作成中にエラー: {e}")
            return {**state, "error": f"サンプルPDF作成エラー: {e}"}
        if not pdf_files: return {**state, "error": f"エラー: フォルダ '{folder_path}' にPDFファイルが見つかりません。"}

    return {**state, "pdf_files": pdf_files, "current_index": 0, "results": [], "error": None}

def select_file_to_process(state: GraphState) -> GraphState:
    print("--- 次のファイルを準備中 ---")
    index = state['current_index']
    pdf_files = state['pdf_files']
    if index < len(pdf_files):
        current_file = pdf_files[index]
        print(f"処理対象: {os.path.basename(current_file)} ({index + 1}/{len(pdf_files)})")
        return {**state, "current_file": current_file, "extracted_text": None, "translated_text": None, "summary": None, "agent_outcome": None, "tool_output": None, "error": None}
    else:
        print("全てのファイルの処理が完了しました。")
        return state # should_continueで判定

def extract_text_from_pdf(state: GraphState) -> GraphState:
    print(f"--- テキスト抽出中: {os.path.basename(state['current_file'])} ---")
    pdf_path = state['current_file']
    if not pdf_path or not os.path.exists(pdf_path): return {**state, "error": f"エラー: ファイルが見つかりません: {pdf_path}"}
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
        print(f"テキスト抽出完了 (約{len(text)}文字)")
        return {**state, "extracted_text": text, "error": None}
    except Exception as e:
        print(f"エラー: テキスト抽出に失敗しました ({pdf_path}): {e}")
        return {**state, "extracted_text": None, "error": f"テキスト抽出エラー: {e}"}

def translate_text(state: GraphState) -> GraphState:
    print("--- テキスト翻訳中 ---")
    text_to_translate = state['extracted_text']
    if not text_to_translate:
        print("翻訳するテキストがありません。スキップします。")
        return {**state, "translated_text": None, "error": "翻訳対象テキストなし"}
    if not llm: return {**state, "error": "LLMが初期化されていません。"}
    messages = [ SystemMessage(content="You are a helpful assistant that translates English text to Japanese."), HumanMessage(content=f"Please translate the following text into natural Japanese:\n\n{text_to_translate}")]
    try:
        response = llm.invoke(messages)
        translated_text = response.content
        print(f"翻訳完了 (約{len(translated_text)}文字)")
        return {**state, "translated_text": translated_text, "error": None}
    except Exception as e:
        print(f"エラー: 翻訳中にエラーが発生しました: {e}")
        return {**state, "translated_text": None, "error": f"翻訳エラー: {e}"}

workflow.add_node("find_pdfs", find_pdf_files)
workflow.add_node("select_file", select_file_to_process)
workflow.add_node("extract_text", extract_text_from_pdf)
workflow.add_node("translate", translate_text)
# --- ここまで前のコードのノード定義関数 ---


# エッジの接続 (更新)
workflow.set_entry_point("find_pdfs")
workflow.add_edge("find_pdfs", "select_file")

# ループ継続判定
workflow.add_conditional_edges(
    "select_file",
    should_continue,
    {
        "continue_processing": "extract_text",
        "end_processing": END
    }
)

# メイン処理フロー
workflow.add_edge("extract_text", "translate")
workflow.add_edge("translate", "summarize")
workflow.add_edge("summarize", "agent_node") # 要約後にエージェントへ

# エージェント後の分岐
workflow.add_conditional_edges(
    "agent_node", # エージェントノードの出力で分岐
    route_after_agent, # 分岐ロジック関数
    {
        "tool_node": "tool_node",    # ツール実行が必要ならToolNodeへ
        "store_result": "store_result" # 不要なら結果保存へ
    }
)

# ツール実行後、結果保存へ
workflow.add_edge("tool_node", "store_result")

# 結果保存後、次のファイル選択へ戻る
workflow.add_edge("store_result", "select_file")


# グラフをコンパイル
app = workflow.compile()


# --- グラフの実行 ---
if __name__ == "__main__":
    pdf_folder = "pdf_files_agent_test" # テスト用フォルダ名変更
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"フォルダ '{pdf_folder}' を作成しました。")
        # サンプルPDF作成は find_pdf_files 内で行われる

    initial_state = {"folder_path": pdf_folder}
    print("\n--- PDF処理パイプライン (エージェント版) 開始 ---")

    # デバッグ用にステップ実行する場合
    # for event in app.stream(initial_state):
    #     for key, value in event.items():
    #         print(f"Node: {key}")
    #         # print(f" State: {value}") # デバッグ用に状態全体を表示
    #         print("-" * 30)
    # final_state = event[key] # 最後のイベントの状態を取得

    # 通常実行
    final_state = app.invoke(initial_state)

    print("\n--- PDF処理パイプライン (エージェント版) 終了 ---")

    # 最終結果の表示
    print("\n--- 最終結果 ---")
    if final_state.get("error") and not final_state.get("results"): # 初期エラーの場合
        print(f"処理開始前にエラーが発生しました: {final_state['error']}")
    elif final_state.get("results"):
        for i, result in enumerate(final_state["results"]):
            print(f"\n[ファイル {i+1}: {os.path.basename(result['file'])}]")
            if result.get('error'): # ファイルごとのエラー
                print(f"  エラー: {result['error']}")
            else:
                print(f"  要約:\n    {result['summary']}")
                if result.get('tool_output'):
                    print(f"  ツール実行結果:\n    {result['tool_output']}")
                else:
                    print("  ツール実行: なし")
    else:
        print("処理結果がありません。")
