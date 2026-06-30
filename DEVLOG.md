## Possible hallucinations

The model said hemoglobin (13.2) and platelets (210) are "higher than normal" — but looking at the test data, both are actually within normal range. A small hallucination worth documenting — this is exactly the kind of governance issue you'll address later with confidence scoring.
Action Items ends with both a real action and "None identified" — a minor formatting quirk in the parser. Not a problem, just noted.

## Summarizer agent
- Built and tested summarizer_agent.py
- Test doc: lab_test.txt (302 chars)
- Summary, key values, and action items all returned correctly
- Minor hallucination: model flagged hemoglobin and platelets as 
  abnormal when both are within normal range — governance risk, 
  flag for confidence scoring in Week 4
- Minor formatting: "None identified" appended after valid action item
- Overall: summarizer working, output quality good ✓