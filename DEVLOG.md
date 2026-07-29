## Possible hallucinations

The model said hemoglobin (13.2) and platelets (210) are "higher than normal" — but looking at the test data, both are actually within normal range. A small hallucination worth documenting — this is exactly the kind of governance issue you'll address later with confidence scoring.
Action Items ends with both a real action and "None identified" — a minor formatting quirk in the parser. Not a problem, just noted.

## June 29, 2026 - Summarizer agent
- Built and tested summarizer_agent.py
- Test doc: lab_test.txt (302 chars)
- Summary, key values, and action items all returned correctly
- Minor hallucination: model flagged hemoglobin and platelets as 
  abnormal when both are within normal range — governance risk, 
  flag for confidence scoring in Week 4
- Minor formatting: "None identified" appended after valid action item
- Overall: summarizer working, output quality good ✓

## June 30, 2026 — Date extraction
- Updated tagger_agent.py with extract_date() function
- First save attempt didn't take — file wasn't actually updated despite paste
- Fixed via git pull, confirmed DATE_PROMPT present, re-ran
- Test result: lab_test.txt → Category: Lab Results, Date: 2026-06-28 ✓

- Updated chat_agent.py: ask_question() now supports filtering by selected_files
- Added get_document_choices() helper for UI population
- Hit a paste error that duplicated/orphaned code outside any function — 
  caught via systematic screenshot review, cleaned up
- Verified with `python -c "import chat_agent"` — clean

Several paste attempts caused duplicated/nested code blocks
- Resolved by full file replacement rather than incremental patching
- Verified clean import
- Added summarize_trend() for multi-document comparison

- file_input now accepts multiple PDFs at once
- handle_upload() loops over file list, tags/archives each
- Document selectors on Summarize/Chat auto-refresh after upload
  (required moving upload_btn.click() below selector definitions)
- Indentation bug caught via VS Code squiggly, fixed

### idea
I'm interested in adding documentation about the general health of someone my age, maybe using NoteBookLM to genereate a markdown file that can be referenced by llama 

## July 19, 2026 - added .md files for reference documentation
Here's what they actually do:
For you right now: PROMPTS.md is your most useful file. It tracks every prompt template in the project, why it was written that way, and what needs to be fixed. When you notice the model giving a wrong answer — like calling a normal lab value "abnormal" — you open PROMPTS.md, find the relevant prompt, update the "Proposed Improvements" section, then go make that change in the actual agent file. It's your maintenance log.
For teaching: when you eventually show this project to non-technical users, agents.md explains what each agent does in plain language. Instead of asking them to read Python code, you hand them that file. It answers "what is this app actually doing?" in terms anyone can follow.
For your AIGP portfolio: these files demonstrate that you thought carefully about governance, failure modes, and improvement cycles — not just that you built something that works. The prompt update log in PROMPTS.md is particularly useful for this, since it shows a deliberate process of testing, observing, and improving over time.
Bottom line: they're like a car's service manual. The car runs fine without the manual sitting in the glovebox — but when something needs fixing, you're glad it's there.

