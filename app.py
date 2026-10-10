"""Cricket Rules RAG Chatbot - Gradio front end.

Answers questions about the Laws of Cricket using only pages crawled from the public web
(see crawl_config.json), with clickable citations, visible refusals, an expandable
retrieval trace (pre- and post-rerank scores), high-stakes referrals and content-age notes.

Run:  pip install -r requirements.txt && python app.py
The index is built on first start from data/processed/ (run the pipeline in README first).
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr

from rag import config
from rag.app_format import format_answer, format_trace
from rag.cli import components
from rag.pipeline import CONFIGS

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
PAGE_TYPES = ["all", "law", "spirit", "pdf", "news", "general"]
RELIABILITY_NOTE = (
    "⚠️ **How far to trust this:** answers come only from the pages listed under *Sources*, "
    "which were captured on the crawl date shown. The bot can still misread a source, and the Laws "
    "are revised from time to time. Check the linked page before relying on an answer, and the "
    "umpires' on-field decision is always final."
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not config.chunks_file("structure").exists():
            raise gr.Error("No processed corpus found. Run the pipeline first (see README: crawl, extract, chunk).")
        _pipeline, _ = components().pipeline(CONFIGS["B3"])
    return _pipeline


def _text(content):
    """Gradio 6 may hand message content back as a list of {"type": "text", "text": ...} parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, (list, tuple)):
        return " ".join(_text(c) for c in content)
    return str(content or "")


def respond(question, history, page_type):
    history = [{"role": m["role"], "content": _text(m["content"])} for m in (history or [])]
    question = (question or "").strip()
    if not question:
        return history, gr.update(), ""
    ans = get_pipeline().answer(question, history, None if page_type == "all" else page_type)
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": format_answer(ans)}]
    return history, format_trace(ans), ""


def create_ui():
    # Gradio 6 moved `theme` to launch() and made the messages format the only one
    blocks_kwargs = {"theme": gr.themes.Soft()} if GRADIO_MAJOR < 6 else {}
    chat_kwargs = {"type": "messages"} if GRADIO_MAJOR < 6 else {}
    with gr.Blocks(title="Cricket Rules RAG Chatbot", **blocks_kwargs) as demo:
        gr.Markdown("# 🏏 Cricket Rules Chatbot\nAsk about the Laws of Cricket. Every answer cites the page it "
                    "came from; if the pages don't cover your question, the bot says so instead of guessing.")
        gr.Markdown(RELIABILITY_NOTE)
        chatbot = gr.Chatbot(label="Conversation", height=460, **chat_kwargs)
        with gr.Row():
            question = gr.Textbox(placeholder="e.g. When is a ball a no ball?", label="Your question",
                                  scale=5, lines=1)
            page_type = gr.Dropdown(PAGE_TYPES, value="all", label="Search only", scale=1)
        with gr.Row():
            ask = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")
        with gr.Accordion("🔍 Retrieval trace (pre- and post-rerank scores)", open=False):
            trace = gr.Markdown("_Ask a question to see which chunks were retrieved._")
        gr.Examples(["How many players are in a cricket team?", "What happens if the ball hits a helmet on the ground?",
                     "And what about a fielder's cap?", "Who won the 2019 World Cup?"], inputs=question)

        ask.click(respond, [question, chatbot, page_type], [chatbot, trace, question])
        question.submit(respond, [question, chatbot, page_type], [chatbot, trace, question])
        clear.click(lambda: ([], "_Ask a question to see which chunks were retrieved._", ""),
                    outputs=[chatbot, trace, question])
    return demo


if __name__ == "__main__":
    on_spaces = "SPACE_ID" in os.environ  # set automatically on Hugging Face Spaces
    launch_kwargs = {"theme": gr.themes.Soft()} if GRADIO_MAJOR >= 6 else {}
    create_ui().queue().launch(
        **launch_kwargs,
        server_name="0.0.0.0",
        # Spaces serves on 7860; leave the port to Gradio there
        server_port=None if on_spaces else int(os.environ.get("PORT", 7861)),
        share=not on_spaces and os.environ.get("GRADIO_SHARE", "1") == "1",
    )
