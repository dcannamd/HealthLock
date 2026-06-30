# app.py
# Gradio UI for the Local AI Health Document Organizer
# Ties together tagger_agent, summarizer_agent, and chat_agent
# with governance, HITL, and multi-document archive features built in.

import gradio as gr
from datetime import datetime
from tagger_agent import tag_document
from summarizer_agent import summarize_document, summarize_trend
from chat_agent import add_document_to_store, ask_question, list_archive_documents, get_document_choices, get_vector_store
from audit_log import log_event

# --- Global state ---
current_vectorstore = None
current_file_path = None
document_history = []  # list of dicts: {filename, category, date, timestamp}

# --- Disclosure banner ---
DISCLOSURE = """
⚠️ **AI Disclosure**
This tool uses a local AI to help you understand your health documents.
It is **not** medical advice. Always verify results with your healthcare provider.
All processing happens on your computer. Nothing is sent to the internet.
"""

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


# --- Step 1: Upload and Tag ---
def handle_upload(files):
    global current_file_path, current_vectorstore, document_history

    if not files:
        return (
            "No file uploaded.", "",
            gr.update(visible=False), format_history(),
            refresh_document_selector(), refresh_document_selector(),
        )

    # Gradio gives a single file object if file_count default, or a list if multiple — normalize to a list
    file_list = files if isinstance(files, list) else [files]

    results = []
    last_category = ""

    for file in file_list:
        current_file_path = file.name
        filename_only = current_file_path.split("/")[-1]

        tag_result = tag_document(current_file_path)
        category = tag_result["category"]
        doc_date = tag_result.get("date", "UNKNOWN")
        last_category = category

        log_event(
            agent="tagger",
            file=current_file_path,
            action="classify",
            result=category,
            confidence="medium",
            flagged=False,
        )

        document_history.insert(0, {
            "filename": filename_only,
            "category": category,
            "date": doc_date,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        # Add this document to the permanent archive
        current_vectorstore = add_document_to_store(
            current_file_path,
            category=category,
            doc_date=doc_date,
        )

        results.append(f"`{filename_only}` (dated {doc_date}) → {category}")

    if len(results) == 1:
        status = f"✅ Document uploaded: {results[0]} — added to archive"
    else:
        status = f"✅ {len(results)} documents uploaded and added to archive:\n\n" + "\n\n".join(results)

    return (
        status,
        last_category,
        gr.update(visible=True),
        format_history(),
        refresh_document_selector(),
        refresh_document_selector(),
    )


def format_history():
    """Render the document history as a readable text block."""
    if not document_history:
        return "No documents processed yet."

    lines = []
    for doc in document_history:
        lines.append(f"📄 {doc['filename']}  →  {doc['category']}  ({doc.get('date', 'UNKNOWN')})  — added {doc['timestamp']}")
    return "\n".join(lines)


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


# --- Document selector refresh (shared logic for Summarize and Chat tabs) ---
def refresh_document_selector():
    """Pull the current archive list and return choices + a sensible default selection."""
    vectorstore = current_vectorstore if current_vectorstore is not None else get_vector_store()
    choices = get_document_choices(vectorstore)

    if not choices:
        return gr.update(choices=[], value=[])

    # Auto-select the newest upload (first item, since get_document_choices sorts newest-first)
    newest_filename = choices[0][1]
    return gr.update(choices=choices, value=[newest_filename])


# --- Step 2: Summarize (single or multi-document trend) ---
def handle_summarize(selected_files):
    if not selected_files:
        return "Please select at least one document.", "", "", "—"

    vectorstore = current_vectorstore if current_vectorstore is not None else get_vector_store()
    all_docs = list_archive_documents(vectorstore)
    doc_lookup = {d["filename"]: d for d in all_docs}

    guardrail = (
        "\n\n---\n"
        "⚠️ *AI summaries can contain errors. "
        "Do not use this as a substitute for professional medical advice.*"
    )

    if len(selected_files) == 1:
        # Single document — individual summary, using the original full-text path
        fname = selected_files[0]
        # Find the original file path from document_history if available, else assume documents/ folder
        matching = [d for d in document_history if d["filename"] == fname]
        file_path = f"documents/{fname}"  # fallback assumption

        result = summarize_document(file_path)

        chars = result["chars_processed"]
        if chars < 200:
            confidence = "🔴 Low"
            note = "Short document — summary may be incomplete."
        elif chars < 1000:
            confidence = "🟡 Medium"
            note = "Moderate detail — review key values carefully."
        else:
            confidence = "🟢 High"
            note = "Good document length — summary should be reliable."

        log_event(
            agent="summarizer",
            file=fname,
            action="summarize",
            result=result["summary"][:100],
            confidence=confidence,
            flagged=False,
        )

        return (
            result["summary"] + guardrail,
            result["key_values"],
            result["action_items"],
            f"{confidence} — {note}",
        )

    else:
        # Multiple documents — trend analysis
        file_paths = [f"documents/{fname}" for fname in selected_files]
        dates = [doc_lookup.get(fname, {}).get("date", "UNKNOWN") for fname in selected_files]

        result = summarize_trend(file_paths, dates)

        log_event(
            agent="summarizer",
            file=", ".join(selected_files),
            action="trend_summarize",
            result=result["trend_summary"][:100],
            confidence="medium",
            flagged=False,
        )

        summary_text = (
            f"**Trend across {len(selected_files)} documents:**\n\n"
            + result["trend_summary"] + guardrail
        )

        return (
            summary_text,
            result["notable_changes"],
            result["discuss_with_doctor"],
            f"🟡 Medium — Trend analysis across {len(selected_files)} documents",
        )


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


# --- Step 3: Chat (filtered by selected documents) ---
def handle_chat(question, history, selected_files):
    vectorstore = current_vectorstore if current_vectorstore is not None else get_vector_store()

    if vectorstore is None:
        return history + [(question, "Please upload a document first.")]

    if not question.strip():
        return history

    if not selected_files:
        return history + [(question, "Please select at least one document to chat about.")]

    answer = ask_question(vectorstore, question, selected_files=selected_files)

    guardrail = (
        "\n\n*This answer is based only on the selected document(s). "
        "Verify important information with your healthcare provider.*"
    )

    log_event(
        agent="chat",
        file=", ".join(selected_files),
        action="question",
        result=question,
        confidence="medium",
        flagged=False,
    )

    return history + [(question, answer + guardrail)]


# --- Build the UI ---
with gr.Blocks(title="HealthLock — Local AI Health Document Organizer") as app:

    gr.Markdown("# 🏥 HealthLock")
    gr.Markdown("### Local AI Health Document Organizer")
    gr.Markdown(DISCLOSURE)

    with gr.Tabs():

        # --- Tab 1: Upload & Tag ---
        with gr.Tab("1️⃣ Upload & Tag"):
            gr.Markdown("### Step 1 — Upload your health document")
            gr.Markdown("*Each document is automatically tagged and added to your permanent local archive.*")
            file_input = gr.File(
                label="Upload PDF or TXT (you can select multiple files)",
                file_types=[".pdf", ".txt"],
                file_count="multiple"
            )
            upload_btn = gr.Button("Upload & Tag", variant="primary")
            upload_status = gr.Markdown()
            category_output = gr.Textbox(
                label="AI-assigned Category",
                interactive=False
            )

            with gr.Row(visible=False) as correction_row:
                gr.Markdown("**Not right? Correct it:**")
                category_dropdown = gr.Dropdown(
                    choices=CATEGORIES,
                    label="Select correct category"
                )
                correct_btn = gr.Button("✏️ Correct Category")
            correction_status = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### 📋 Document History")
            history_display = gr.Textbox(
                label="Processed Documents",
                lines=8,
                interactive=False,
                value="No documents processed yet."
            )

         

        # --- Tab 2: Summarize ---
        with gr.Tab("2️⃣ Summarize"):
            gr.Markdown("### Step 2 — Get a plain-language summary")
            gr.Markdown("*Select one document for an individual summary, or multiple documents to see trends over time.*")

            summarize_doc_selector = gr.CheckboxGroup(
                label="Select document(s) to summarize",
                choices=[],
            )
            refresh_selector_btn = gr.Button("🔄 Refresh Document List")

            summarize_btn = gr.Button("Summarize Selected", variant="primary")
            confidence_output = gr.Textbox(
                label="Confidence Level",
                interactive=False
            )
            summary_output = gr.Textbox(
                label="Summary",
                lines=6,
                interactive=False
            )
            key_values_output = gr.Textbox(
                label="Key Values / Notable Changes",
                lines=6,
                interactive=False
            )
            action_items_output = gr.Textbox(
                label="Action Items / Things to Discuss with Doctor",
                lines=4,
                interactive=False
            )

            flag_btn = gr.Button("🚩 Flag This Summary for Review")
            flag_status = gr.Markdown()

            refresh_selector_btn.click(
                refresh_document_selector,
                outputs=[summarize_doc_selector]
            )
            summarize_btn.click(
                handle_summarize,
                inputs=[summarize_doc_selector],
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
            gr.Markdown("### Step 3 — Ask questions about your documents")
            gr.Markdown(
                "*Select which document(s) to include, then ask questions. "
                "Nothing is sent to the internet.*"
            )

            chat_doc_selector = gr.CheckboxGroup(
                label="Select document(s) to include in chat",
                choices=[],
            )
            refresh_chat_selector_btn = gr.Button("🔄 Refresh Document List")

            chatbot = gr.Chatbot(label="Document Chat")
            chat_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Is my cholesterol improving over time?"
            )
            chat_btn = gr.Button("Ask", variant="primary")

            refresh_chat_selector_btn.click(
                refresh_document_selector,
                outputs=[chat_doc_selector]
            )
            chat_btn.click(
                handle_chat,
                inputs=[chat_input, chatbot, chat_doc_selector],
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

    upload_btn.click(
        handle_upload,
        inputs=[file_input],
        outputs=[
            upload_status,
            category_output,
            correction_row,
            history_display,
            summarize_doc_selector,
            chat_doc_selector,
        ]
    )
    correct_btn.click(
        correct_category,
        inputs=[category_dropdown],
        outputs=[correction_status]
    )

if __name__ == "__main__":
    app.launch()