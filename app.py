# app.py
# Gradio UI for HealthLock — Local AI Health Document Organizer
#
# Ties together tagger_agent, summarizer_agent, and chat_agent with
# governance, HITL, and multi-document archive features.
#
# Changes in this version:
#   - Per-document progress feedback during upload (gr.Progress)
#   - Document history and selectors populate on page load (app.load)
#   - Documents can be deleted from both disk and the vector archive
#   - Richer document labels showing provenance (AI-assigned vs human-confirmed)
#   - Human category corrections now persist to history, not just the audit log

import gradio as gr
import json
import shutil
from datetime import datetime
from pathlib import Path

from tagger_agent import tag_document
from summarizer_agent import summarize_document, summarize_trend
from chat_agent import (
    add_document_to_store,
    ask_question,
    list_archive_documents,
    get_document_choices,
    get_vector_store,
)
from audit_log import log_event

# --- Global state ---
current_vectorstore = None
current_file_path = None
document_history = []  # [{filename, full_path, category, date, timestamp, corrected}]

HISTORY_FILE = "logs/document_history.json"
DOCUMENTS_DIR = "documents"

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

DISCLOSURE = """
⚠️ **AI Disclosure**
This tool uses a local AI to help you understand your health documents.
It is **not** medical advice. Always verify results with your healthcare provider.
All processing happens on your computer. Nothing is sent to the internet.
"""


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

def save_history():
    """Write document_history to disk so it survives app restarts."""
    Path("logs").mkdir(exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(document_history, f, indent=2)


def load_history():
    """Read document_history from disk on startup."""
    global document_history
    try:
        with open(HISTORY_FILE, "r") as f:
            document_history = json.load(f)
        print(f"Loaded {len(document_history)} documents from history.")
    except (FileNotFoundError, json.JSONDecodeError):
        document_history = []


load_history()


def history_entry(filename):
    """Return the most recent history entry for a filename, or None."""
    for d in document_history:
        if d.get("filename") == filename:
            return d
    return None


def path_for(filename):
    """Return the on-disk path for a filename, or None if unknown."""
    entry = history_entry(filename)
    return entry.get("full_path") if entry else None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_history():
    """Render document history as a readable text block, newest first."""
    if not document_history:
        return "No documents yet. Upload a PDF or TXT file to get started."

    lines = []
    for doc in document_history:
        provenance = "you confirmed" if doc.get("corrected") else "AI-assigned"
        date = doc.get("date", "UNKNOWN")
        if date == "UNKNOWN":
            date = "no date found"
        lines.append(
            f"{doc['filename']}\n"
            f"    {doc['category']} ({provenance})  ·  {date}  "
            f"·  added {doc['timestamp']}"
        )
    return "\n\n".join(lines)


def build_choices():
    """
    Build selector choices from history (authoritative for provenance) with
    a fallback to the vector archive for anything history doesn't know about.
    """
    labels = []
    seen = set()

    for doc in document_history:
        fname = doc.get("filename")
        if not fname or fname in seen or fname == "unknown":
            continue
        seen.add(fname)
        date = doc.get("date", "UNKNOWN")
        date_txt = "no date" if date == "UNKNOWN" else date
        provenance = "confirmed" if doc.get("corrected") else "AI"
        labels.append((
            f"{fname}  —  {doc.get('category', 'Unknown')} ({provenance})  —  {date_txt}",
            fname,
        ))

    # Anything in the archive but not in history (e.g. from an older session)
    try:
        vectorstore = current_vectorstore or get_vector_store()
        for label, fname in get_document_choices(vectorstore):
            if fname not in seen and fname != "unknown":
                seen.add(fname)
                labels.append((f"{label}  —  not in history", fname))
    except Exception as e:
        print(f"Could not read archive for selector: {e}")

    return labels


def refresh_document_selector():
    """Populate a selector, pre-checking the newest document."""
    choices = build_choices()
    if not choices:
        return gr.update(choices=[], value=[])
    return gr.update(choices=choices, value=[choices[0][1]])


def refresh_delete_selector():
    """Populate the delete dropdown. Nothing is pre-selected, deliberately."""
    choices = build_choices()
    return gr.update(choices=choices, value=None)


def populate_on_load():
    """Fill history and all selectors when the page first loads."""
    return (
        format_history(),
        refresh_document_selector(),
        refresh_document_selector(),
        refresh_delete_selector(),
    )


# ---------------------------------------------------------------------------
# Step 1 — Upload and tag
# ---------------------------------------------------------------------------

def handle_upload(files, progress=gr.Progress()):
    """
    Copy each uploaded file into documents/, tag it, and add it to the
    persistent archive. Reports progress per document.
    """
    global current_file_path, current_vectorstore, document_history

    if not files:
        return (
            "No file selected.",
            "",
            gr.update(visible=False),
            format_history(),
            refresh_document_selector(),
            refresh_document_selector(),
            refresh_delete_selector(),
        )

    file_list = files if isinstance(files, list) else [files]
    total = len(file_list)

    Path(DOCUMENTS_DIR).mkdir(exist_ok=True)

    results = []
    last_category = ""

    progress(0, desc=f"Starting — {total} document(s) to process")

    for i, file in enumerate(file_list, start=1):
        src_path = file.name
        filename_only = Path(src_path).name

        progress(
            (i - 1) / total,
            desc=f"[{i}/{total}] Reading {filename_only}",
        )

        dest_path = str(Path(DOCUMENTS_DIR) / filename_only)
        try:
            # Skip the copy if the source already is the destination
            if Path(src_path).resolve() != Path(dest_path).resolve():
                shutil.copy2(src_path, dest_path)
        except Exception as e:
            results.append(f"{filename_only} — could not be saved: {e}")
            continue

        current_file_path = dest_path

        progress(
            (i - 0.6) / total,
            desc=f"[{i}/{total}] Categorising {filename_only} — this takes 10-30s",
        )

        try:
            tag_result = tag_document(current_file_path)
        except Exception as e:
            results.append(f"{filename_only} — could not be read: {e}")
            continue

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
            "full_path": dest_path,
            "category": category,
            "date": doc_date,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "corrected": False,
        })

        progress(
            (i - 0.3) / total,
            desc=f"[{i}/{total}] Indexing {filename_only} for search",
        )

        try:
            current_vectorstore = add_document_to_store(
                current_file_path,
                category=category,
                doc_date=doc_date,
            )
        except Exception as e:
            results.append(f"{filename_only} — tagged, but indexing failed: {e}")
            continue

        date_txt = "no date found" if doc_date == "UNKNOWN" else doc_date
        results.append(f"{filename_only} — {category}, dated {date_txt}")

        progress(i / total, desc=f"[{i}/{total}] Done — {filename_only}")

    save_history()

    if len(results) == 1:
        status = f"**Added:** {results[0]}"
    else:
        status = (
            f"**Added {len(results)} document(s):**\n\n"
            + "\n\n".join(f"- {r}" for r in results)
        )

    return (
        status,
        last_category,
        gr.update(visible=True),
        format_history(),
        refresh_document_selector(),
        refresh_document_selector(),
        refresh_delete_selector(),
    )


def correct_category(new_category):
    """Human override of the AI's category. Persists to history and the log."""
    if not new_category:
        return "Select a category first.", format_history(), refresh_document_selector(), refresh_document_selector(), refresh_delete_selector()

    if not current_file_path:
        return "Upload a document first.", format_history(), refresh_document_selector(), refresh_document_selector(), refresh_delete_selector()

    fname = Path(current_file_path).name
    entry = history_entry(fname)
    if entry:
        entry["category"] = new_category
        entry["corrected"] = True
        save_history()

    log_event(
        agent="tagger",
        file=current_file_path,
        action="human_correction",
        result=new_category,
        confidence="high",
        flagged=False,
    )

    return (
        f"Category for **{fname}** changed to **{new_category}**. "
        "This is now marked as confirmed by you.",
        format_history(),
        refresh_document_selector(),
        refresh_document_selector(),
        refresh_delete_selector(),
    )


# ---------------------------------------------------------------------------
# Delete a document
# ---------------------------------------------------------------------------

def delete_document(filename, confirm):
    """
    Remove a document from the vector archive, from history, and from disk.
    Requires an explicit confirmation checkbox.
    """
    global document_history, current_file_path

    if not filename:
        return ("Select a document to delete.", format_history(),
                refresh_document_selector(), refresh_document_selector(),
                refresh_delete_selector())

    if not confirm:
        return ("Tick the confirmation box to delete. This cannot be undone.",
                format_history(), refresh_document_selector(),
                refresh_document_selector(), refresh_delete_selector())

    notes = []

    # 1. Remove chunks from the vector archive
    try:
        vectorstore = current_vectorstore or get_vector_store()
        data = vectorstore.get(where={"source_file": filename})
        ids = data.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)
            notes.append(f"removed {len(ids)} indexed chunk(s)")
        else:
            notes.append("no indexed chunks found")
    except Exception as e:
        notes.append(f"archive removal failed: {e}")

    # 2. Remove the file from disk
    file_path = path_for(filename) or str(Path(DOCUMENTS_DIR) / filename)
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            notes.append("deleted file from documents/")
        else:
            notes.append("file was not on disk")
    except Exception as e:
        notes.append(f"file deletion failed: {e}")

    # 3. Remove from history
    before = len(document_history)
    document_history = [d for d in document_history if d.get("filename") != filename]
    if len(document_history) < before:
        notes.append("removed from history")
    save_history()

    if current_file_path and Path(current_file_path).name == filename:
        current_file_path = None

    log_event(
        agent="system",
        file=filename,
        action="delete_document",
        result="; ".join(notes),
        confidence="—",
        flagged=False,
    )

    return (
        f"**Deleted {filename}** — {'; '.join(notes)}.",
        format_history(),
        refresh_document_selector(),
        refresh_document_selector(),
        refresh_delete_selector(),
    )


# ---------------------------------------------------------------------------
# Step 2 — Summarize (single document or multi-document trend)
# ---------------------------------------------------------------------------

def handle_summarize(selected_files, progress=gr.Progress()):
    if not selected_files:
        return "Select at least one document.", "", "", "—"

    guardrail = (
        "\n\n---\n"
        "*AI summaries can contain errors. This is not medical advice — "
        "verify anything important with your healthcare provider.*"
    )

    # Resolve filenames to real paths
    resolved = []
    missing = []
    for fname in selected_files:
        p = path_for(fname)
        if p and Path(p).exists():
            resolved.append((fname, p))
        else:
            missing.append(fname)

    if missing and not resolved:
        return (
            "Could not find the original file(s) for: "
            + ", ".join(missing)
            + ". They may have been deleted. Re-upload to summarize them.",
            "", "", "—",
        )

    note_missing = ""
    if missing:
        note_missing = (
            "\n\n*Skipped (file not found): " + ", ".join(missing) + "*"
        )

    # --- Single document ---
    if len(resolved) == 1:
        fname, file_path = resolved[0]
        progress(0.1, desc=f"Reading {fname}")
        progress(0.3, desc="Summarising — this takes 20-60s on this computer")

        try:
            result = summarize_document(file_path)
        except Exception as e:
            return f"Could not summarize {fname}: {e}", "", "", "—"

        progress(0.9, desc="Formatting result")

        chars = result["chars_processed"]
        if chars < 200:
            confidence, note = "Low", "Very short document — summary may be incomplete."
        elif chars < 1000:
            confidence, note = "Medium", "Moderate detail — check key values against the original."
        else:
            confidence, note = "High", "Good document length, but still verify key numbers."

        log_event(
            agent="summarizer",
            file=fname,
            action="summarize",
            result=result["summary"][:100],
            confidence=confidence,
            flagged=False,
        )

        header = f"**{fname}**\n\n"
        return (
            header + result["summary"] + guardrail + note_missing,
            result["key_values"],
            result["action_items"],
            f"{confidence} — {note}",
        )

    # --- Multiple documents: trend ---
    file_paths = [p for _, p in resolved]
    fnames = [f for f, _ in resolved]

    dates = []
    for fname in fnames:
        entry = history_entry(fname)
        dates.append(entry.get("date", "UNKNOWN") if entry else "UNKNOWN")

    est = "1-2 minutes" if len(file_paths) <= 5 else "2-4 minutes"
    progress(0.1, desc=f"Reading {len(file_paths)} documents")
    progress(0.3, desc=f"Comparing across dates — this takes about {est}")

    try:
        result = summarize_trend(file_paths, dates)
    except Exception as e:
        return f"Could not compare these documents: {e}", "", "", "—"

    progress(0.9, desc="Formatting result")

    log_event(
        agent="summarizer",
        file=", ".join(fnames),
        action="trend_summarize",
        result=result["trend_summary"][:100],
        confidence="medium",
        flagged=False,
    )

    unknown_dates = sum(1 for d in dates if d == "UNKNOWN")
    caveats = []
    if unknown_dates:
        caveats.append(
            f"{unknown_dates} document(s) had no readable date, so their "
            "position in the timeline is a guess."
        )
    caveat_txt = ("\n\n*" + " ".join(caveats) + "*") if caveats else ""

    listing = "\n".join(
        f"- {d if d != 'UNKNOWN' else 'no date'} · {f}"
        for f, d in sorted(zip(fnames, dates), key=lambda x: x[1])
    )

    summary_text = (
        f"**Comparing {len(fnames)} documents**\n\n{listing}\n\n---\n\n"
        + result["trend_summary"]
        + caveat_txt
        + guardrail
        + note_missing
    )

    return (
        summary_text,
        result["notable_changes"],
        result["discuss_with_doctor"],
        f"Medium — comparison across {len(fnames)} documents",
    )


def flag_for_review(section):
    if not section or not section.strip():
        return "Nothing to flag yet — generate a summary first."

    log_event(
        agent="summarizer",
        file=Path(current_file_path).name if current_file_path else "multiple",
        action="flagged_for_review",
        result=section[:200],
        confidence="—",
        flagged=True,
    )
    return (
        "Flagged for review. This is recorded in your activity log — "
        "filter by `flagged_for_review` to find it again."
    )


# ---------------------------------------------------------------------------
# Step 3 — Chat
# ---------------------------------------------------------------------------

def handle_chat(question, history, selected_files, progress=gr.Progress()):
    if not question or not question.strip():
        return history, ""

    if not selected_files:
        return history + [(
            question,
            "Select at least one document above before asking a question."
        )], ""

    try:
        vectorstore = current_vectorstore or get_vector_store()
    except Exception as e:
        return history + [(question, f"Could not open your document archive: {e}")], ""

    progress(0.2, desc="Searching your documents")
    progress(0.5, desc="Answering — this takes 20-60s on this computer")

    try:
        answer = ask_question(vectorstore, question, selected_files=selected_files)
    except Exception as e:
        return history + [(question, f"Something went wrong answering that: {e}")], ""

    scope = (
        f"1 document" if len(selected_files) == 1
        else f"{len(selected_files)} documents"
    )
    guardrail = (
        f"\n\n---\n*Answered using {scope} you selected. "
        "Verify anything important with your healthcare provider.*"
    )

    log_event(
        agent="chat",
        file=", ".join(selected_files),
        action="question",
        result=question,
        confidence="medium",
        flagged=False,
    )

    return history + [(question, answer + guardrail)], ""


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def refresh_log(action_filter):
    try:
        with open("logs/audit_log.json", "r") as f:
            entries = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return "No activity recorded yet."
    except json.JSONDecodeError:
        return "The activity log could not be read — it may be corrupted."

    if action_filter and action_filter != "All actions":
        if action_filter == "Flagged for review only":
            entries = [e for e in entries if e.get("flagged_for_review")]
        else:
            entries = [e for e in entries if e.get("action") == action_filter]

    if not entries:
        return "No entries match that filter."

    entries.reverse()  # newest first

    lines = []
    for e in entries[:200]:
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        flag = "  [FLAGGED]" if e.get("flagged_for_review") else ""
        fname = Path(e.get("file", "")).name or e.get("file", "")
        lines.append(
            f"{ts}  ·  {e.get('agent', '')}  ·  {e.get('action', '')}{flag}\n"
            f"    {fname}\n"
            f"    {str(e.get('result', ''))[:160]}"
        )

    header = f"Showing {min(len(entries), 200)} of {len(entries)} entries, newest first.\n\n"
    return header + "\n\n".join(lines)


def export_log():
    import csv
    try:
        with open("logs/audit_log.json", "r") as f:
            entries = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return "No activity log to export yet."

    if not entries:
        return "The activity log is empty."

    fieldnames = sorted({k for e in entries for k in e.keys()})
    with open("logs/audit_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    return f"Exported {len(entries)} entries to `logs/audit_log.csv`."


ACTION_FILTERS = [
    "All actions",
    "Flagged for review only",
    "classify",
    "human_correction",
    "summarize",
    "trend_summarize",
    "question",
    "delete_document",
]


# ---------------------------------------------------------------------------
# Where your data lives
# ---------------------------------------------------------------------------

def data_location_report():
    cwd = Path.cwd()
    docs = cwd / DOCUMENTS_DIR
    chroma = cwd / "chroma_db"
    logs = cwd / "logs"

    def describe(p, label):
        if not p.exists():
            return f"- **{label}**: `{p}` — does not exist yet"
        if p.is_dir():
            n = len(list(p.iterdir()))
            return f"- **{label}**: `{p}` — {n} item(s)"
        return f"- **{label}**: `{p}`"

    lines = [
        "Everything HealthLock stores is in these folders on this computer:",
        "",
        describe(docs, "Your documents"),
        describe(chroma, "Search index"),
        describe(logs, "Activity log and history"),
        "",
        "To delete all of your data, delete those three folders.",
        "",
        "**Verifying the privacy claim.** Run `python privacy_check.py` in a "
        "second terminal while using the app — it watches for network "
        "connections and writes a dated report to `logs/privacy_report.json`. "
        "The simplest test is stronger though: turn off Wi-Fi and confirm "
        "everything still works.",
        "",
        "**Disk encryption.** These files are only as protected as the disk "
        "they sit on. On a Mac, check System Settings → Privacy & Security → "
        "FileVault.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.hl-mono textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
                    font-size: 12.5px !important; line-height: 1.6 !important; }
.hl-footer { color: #666; font-size: 12px; font-family: ui-monospace, Menlo, monospace; }
"""

with gr.Blocks(title="HealthLock — Local AI Health Document Organizer", css=CSS) as app:

    gr.Markdown("# HealthLock")
    gr.Markdown("**Local AI Health Document Organizer** — runs on this computer only")
    gr.Markdown(DISCLOSURE)

    with gr.Tabs():

        # ---------------- Tab 1: Archive ----------------
        with gr.Tab("1 · Your archive"):
            gr.Markdown("### Add documents")
            gr.Markdown(
                "Each file is copied into your `documents/` folder, read by the "
                "local AI to guess its type and date, then indexed so you can "
                "search it. Expect roughly 20-40 seconds per document."
            )

            file_input = gr.File(
                label="PDF or TXT — you can select several at once",
                file_types=[".pdf", ".txt"],
                file_count="multiple",
            )
            upload_btn = gr.Button("Add documents", variant="primary")
            upload_status = gr.Markdown()

            category_output = gr.Textbox(
                label="Category assigned to the last document",
                interactive=False,
            )

            with gr.Row(visible=False) as correction_row:
                category_dropdown = gr.Dropdown(
                    choices=CATEGORIES,
                    label="Wrong? Pick the right category",
                )
                correct_btn = gr.Button("Change category")
            correction_status = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### Your documents")
            history_display = gr.Textbox(
                label="Newest first",
                lines=14,
                interactive=False,
                elem_classes=["hl-mono"],
            )

            gr.Markdown("---")
            gr.Markdown("### Delete a document")
            gr.Markdown(
                "Removes the file from `documents/`, removes it from the search "
                "index, and removes it from your history. This cannot be undone."
            )
            delete_selector = gr.Dropdown(
                choices=[], label="Document to delete", value=None
            )
            delete_confirm = gr.Checkbox(
                label="Yes, permanently delete this document", value=False
            )
            delete_btn = gr.Button("Delete document", variant="stop")
            delete_status = gr.Markdown()

        # ---------------- Tab 2: Summarize ----------------
        with gr.Tab("2 · Summarise or compare"):
            gr.Markdown("### One document, or several")
            gr.Markdown(
                "**Tick one document** for a plain-language summary of it.\n\n"
                "**Tick two or more** to compare them over time and see what "
                "changed. These are two different jobs with different output."
            )

            summarize_doc_selector = gr.CheckboxGroup(
                label="Documents", choices=[]
            )
            refresh_selector_btn = gr.Button("Reload document list")

            summarize_btn = gr.Button("Summarise / compare", variant="primary")

            confidence_output = gr.Textbox(
                label="How much to trust this", interactive=False
            )
            summary_output = gr.Textbox(
                label="Summary", lines=12, interactive=False
            )
            key_values_output = gr.Textbox(
                label="Key values (one doc) · What changed (several)",
                lines=10, interactive=False, elem_classes=["hl-mono"],
            )
            action_items_output = gr.Textbox(
                label="Next steps · Things to raise with your doctor",
                lines=6, interactive=False,
            )

            flag_btn = gr.Button("Flag this for review")
            flag_status = gr.Markdown()

        # ---------------- Tab 3: Chat ----------------
        with gr.Tab("3 · Ask a question"):
            gr.Markdown("### Ask about your documents")
            gr.Markdown(
                "Answers come only from the documents you tick. If the answer "
                "isn't in them, the AI will say so rather than guess — that's "
                "working correctly, not an error."
            )

            chat_doc_selector = gr.CheckboxGroup(
                label="Documents in scope for this conversation", choices=[]
            )
            refresh_chat_selector_btn = gr.Button("Reload document list")

            chatbot = gr.Chatbot(label="Conversation", height=420)
            chat_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Did my cholesterol change between 2020 and 2026?",
            )
            chat_btn = gr.Button("Ask", variant="primary")

        # ---------------- Tab 4: Activity log ----------------
        with gr.Tab("4 · Activity log"):
            gr.Markdown("### Every AI action on this computer")
            gr.Markdown(
                "This log never leaves your machine. It records what ran, on "
                "which document, and what came back."
            )
            log_filter = gr.Dropdown(
                choices=ACTION_FILTERS, value="All actions", label="Show"
            )
            audit_display = gr.Textbox(
                label="Activity", lines=22, interactive=False,
                elem_classes=["hl-mono"],
            )
            with gr.Row():
                refresh_btn = gr.Button("Reload log")
                export_btn = gr.Button("Export as CSV")
            export_status = gr.Markdown()

        # ---------------- Tab 5: Where your data lives ----------------
        with gr.Tab("5 · Where your data lives"):
            gr.Markdown("### Your data on this computer")
            data_location_display = gr.Markdown()
            check_location_btn = gr.Button("Check folders")

    gr.Markdown(
        "Model: Llama 3 via Ollama, running on this machine · "
        "Nothing is sent to the internet",
        elem_classes=["hl-footer"],
    )

    # --- Wiring (after all components exist) ---

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
            delete_selector,
        ],
    )

    correct_btn.click(
        correct_category,
        inputs=[category_dropdown],
        outputs=[
            correction_status,
            history_display,
            summarize_doc_selector,
            chat_doc_selector,
            delete_selector,
        ],
    )

    delete_btn.click(
        delete_document,
        inputs=[delete_selector, delete_confirm],
        outputs=[
            delete_status,
            history_display,
            summarize_doc_selector,
            chat_doc_selector,
            delete_selector,
        ],
    )

    refresh_selector_btn.click(
        refresh_document_selector, outputs=[summarize_doc_selector]
    )
    refresh_chat_selector_btn.click(
        refresh_document_selector, outputs=[chat_doc_selector]
    )

    summarize_btn.click(
        handle_summarize,
        inputs=[summarize_doc_selector],
        outputs=[
            summary_output,
            key_values_output,
            action_items_output,
            confidence_output,
        ],
    )
    flag_btn.click(flag_for_review, inputs=[summary_output], outputs=[flag_status])

    chat_btn.click(
        handle_chat,
        inputs=[chat_input, chatbot, chat_doc_selector],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        handle_chat,
        inputs=[chat_input, chatbot, chat_doc_selector],
        outputs=[chatbot, chat_input],
    )

    refresh_btn.click(refresh_log, inputs=[log_filter], outputs=[audit_display])
    log_filter.change(refresh_log, inputs=[log_filter], outputs=[audit_display])
    export_btn.click(export_log, outputs=[export_status])

    check_location_btn.click(data_location_report, outputs=[data_location_display])

    # Populate everything when the page loads
    app.load(
        populate_on_load,
        outputs=[
            history_display,
            summarize_doc_selector,
            chat_doc_selector,
            delete_selector,
        ],
    )
    app.load(refresh_log, inputs=[log_filter], outputs=[audit_display])
    app.load(data_location_report, outputs=[data_location_display])


if __name__ == "__main__":
    app.queue()
    app.launch()