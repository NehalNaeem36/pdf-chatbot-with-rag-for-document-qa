from __future__ import annotations


def answer_question(pdf_file, question: str) -> tuple[str, str]:
    if pdf_file is None:
        return "Upload a PDF first.", ""
    if not question.strip():
        return "Enter a question.", ""

    # This is a placeholder until the retrieval and QA pipeline is wired together.
    return "UI skeleton ready. Backend wiring comes next.", "Citation: pending"


def build_app() -> gr.Blocks:
    import gradio as gr

    with gr.Blocks(title="PDF QA") as demo:
        gr.Markdown("# PDF QA")
        gr.Markdown("Upload one text-based PDF and ask a question about it.")

        pdf_file = gr.File(label="PDF file", file_types=[".pdf"])
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        citation = gr.Textbox(label="Citation")
        ask_button = gr.Button("Ask")

        ask_button.click(
            fn=answer_question,
            inputs=[pdf_file, question],
            outputs=[answer, citation],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
