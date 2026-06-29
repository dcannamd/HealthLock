# app.py
# Gradio UI for the Local AI Health Document Organizer
# Ties together tagger_agent, summarizer_agent, and chat_agent
# with governance and HITL features built in.

import gradio as gr
from tagger_agent import tag_document
from summarizer_agent import summarize_document
from chat_agent import build_vector_store, ask_question
from audit_log import log_event

# --- Global state ---
current_vectorstore = None
current_file_path = None

# --- Disclosure banner ---
DISCLOSURE = """
⚠️ **AI Disclosure**
This tool uses a local AI to help you understand your health documents.
It is **not** medical advice. Always verify results with your healthcare provider.
All processing happens on your computer. Nothing is sent to the internet.
"""

# --- Step 1: Upload and Tag ---
def handle_upload(file):
    global current_file_path, current_vectorstore

    if file is None:
        return "No file uploaded.", "", "", "", gr.update(visible=False)

    current_file_path = file.name

    # Tag the document
    tag_result = tag_document(current_file_path)
    category = tag_result["category"]

    # Log the event
    log_event(
        agent="tagger",
        file=current_file_path,
        action="classify",
        result=category,
        confidence="medium",
        flagged=False,
    )

    return (
        f"✅ Document uploaded: `{current_file_path}`",
        category,
        "",   # clear summary
        "",   # clear action items
        gr.update(visible=True),  # show correct button
    )


# --- Human override: correct the category ---
def correct_category(new_category):
    log_event(
        agent="tagger",
        file=current_file_path,
        action="human_correction",
        result=new_category,
        confidence="high",
        flagged=False,
    )
    return f"✅ Category corrected to: **{new_category}**"


# --- Step 2: Summarize ---
def handle_summarize():
    global current_vectorstore

    if current_file_path is None:
        return "Please upload a document first.", "", "", "—"

    result = summarize_document(current_file_path)

    # Build vector store for chat
    current_vectorstore = build_vector_store(current_file_path)

    # Simple confidence scoring based on document length
    chars = result["chars_processed"]
    if chars < 200:
        confidence = "🔴 Low"
        confidence_note = "Short document — summary may be incomplete."
    elif chars < 1000:
        confidence = "🟡 Medium"
        confidence_note = "Moderate detail — review key values carefully."
    else:
        confidence = "🟢 High"
        confidence_note = "Good document length — summary should be reliable."

    log_event(
        agent="summarizer",
        file=current_file_path,
        action="summarize",
        result=result["summary"][:100],
        confidence=confidence,
        flagged=False,
    )

    guardrail = (
        "\n\n---\n"
        "⚠️ *AI summaries can contain errors. "
        "Do not use this as a substitute for professional medical advice.*"
    )

    return (
        result["summary"] + guardrail,
        result["key_values"],
        result["action_items"],
        f"{confidence} — {confidence_note}",
    )


# --- Flag for human review ---
def flag_for_review(section):
    log_event(
        agent="summarizer",
        file=current_file_path,
        action="flagged_for_review",
        result=section,
        confidence="—",
        flagged=True,
    )
    return "🚩 Flagged for review. This has been recorded in your audit log."


# --- Step 3: Chat ---
def handle_chat(question, history):
    if current_vectorstore is None:
        return history + [
            (question, "Please upload and summarize a document first.")
        ]

    if not question.strip():
        return history

    answer = ask_question(current_vectorstore, question)

    guardrail = (
        "\n\n*This answer is based only on the documents you uploaded. "
        "Verify important information with your healthcare provider.*"
    )

    log_event(
        agent="chat",
        file=current_file_path,
        action="question",
        result=question,
        confidence="medium",
        flagged=False,
    )

    return history + [(question, answer + guardrail)]


# --- Build the UI ---
CATEGORIES = [
    "Lab Results",
    "Prescription",
    "Appointment Summary",
    "Insurance / EOB",
    "Medical History",
    "Imaging Report",
    "Referral",
    "Other",
]

with gr.Blocks(title="HealthLock — Local AI Health Document Organizer") as app:

    gr.Markdown("# 🏥 HealthLock")
    gr.Markdown("### Local AI Health Document Organizer")
    gr.Markdown(DISCLOSURE)

    with gr.Tabs():

        # --- Tab 1: Upload & Tag ---
        with gr.Tab("1️⃣ Upload & Tag"):
            gr.Markdown("### Step 1 — Upload your health document")
            file_input = gr.File(
                label="Upload PDF or TXT",
                file_types=[".pdf", ".txt"]
            )
            upload_btn = gr.Button("Upload & Tag", variant="primary")
            upload_status = gr.Markdown()
            category_output = gr.Textbox(
                label="AI-assigned Category",
                interactive=False
            )

            # HITL: Human correction
            with gr.Row(visible=False) as correction_row:
                gr.Markdown("**Not right? Correct it:**")
                category_dropdown = gr.Dropdown(
                    choices=CATEGORIES,
                    label="Select correct category"
                )
                correct_btn = gr.Button("✏️ Correct Category")
            correction_status = gr.Markdown()

            upload_btn.click(
                handle_upload,
                inputs=[file_input],
                outputs=[
                    upload_status,
                    category_output,
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    correction_row,
                ]
            )
            correct_btn.click(
                correct_category,
                inputs=[category_dropdown],
                outputs=[correction_status]
            )

        # --- Tab 2: Summarize ---
        with gr.Tab("2️⃣ Summarize"):
            gr.Markdown("### Step 2 — Get a plain-language summary")
            summarize_btn = gr.Button("Summarize Document", variant="primary")
            confidence_output = gr.Textbox(
                label="Confidence Level",
                interactive=False
            )
            summary_output = gr.Textbox(
                label="Summary",
                lines=5,
                interactive=False
            )
            key_values_output = gr.Textbox(
                label="Key Values",
                lines=5,
                interactive=False
            )
            action_items_output = gr.Textbox(
                label="Action Items",
                lines=3,
                interactive=False
            )

            # HITL: Flag for review
            flag_btn = gr.Button("🚩 Flag This Summary for Review")
            flag_status = gr.Markdown()

            summarize_btn.click(
                handle_summarize,
                outputs=[
                    summary_output,
                    key_values_output,
                    action_items_output,
                    confidence_output,
                ]
            )
            flag_btn.click(
                flag_for_review,
                inputs=[summary_output],
                outputs=[flag_status]
            )

        # --- Tab 3: Chat ---
        with gr.Tab("3️⃣ Chat"):
            gr.Markdown("### Step 3 — Ask questions about your document")
            gr.Markdown(
                "*Questions are answered using only your uploaded document. "
                "Nothing is sent to the internet.*"
            )
            chatbot = gr.Chatbot(label="Document Chat")
            chat_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. What was my cholesterol level?"
            )
            chat_btn = gr.Button("Ask", variant="primary")

            chat_btn.click(
                handle_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot]
            )

        # --- Tab 4: Audit Log ---
        with gr.Tab("📋 Audit Log"):
            gr.Markdown("### Your local audit log")
            gr.Markdown(
                "Every AI action is recorded here. "
                "This log never leaves your computer."
            )
            audit_display = gr.Textbox(
                label="Audit Log",
                lines=20,
                interactive=False
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Log")
                export_btn = gr.Button("📥 Export as CSV")
            export_status = gr.Markdown()

            def refresh_log():
                try:
                    with open("logs/audit_log.json", "r") as f:
                        return f.read()
                except FileNotFoundError:
                    return "No audit log entries yet."

            def export_log():
                import json
                import csv
                try:
                    with open("logs/audit_log.json", "r") as f:
                        entries = [
                            json.loads(line)
                            for line in f
                            if line.strip()
                        ]
                    with open("logs/audit_log.csv", "w", newline="") as f:
                        if entries:
                            writer = csv.DictWriter(
                                f, fieldnames=entries[0].keys()
                            )
                            writer.writeheader()
                            writer.writerows(entries)
                    return "✅ Exported to logs/audit_log.csv"
                except FileNotFoundError:
                    return "No audit log to export yet."

            refresh_btn.click(refresh_log, outputs=[audit_display])
            export_btn.click(export_log, outputs=[export_status])

if __name__ == "__main__":
    app.launch()
