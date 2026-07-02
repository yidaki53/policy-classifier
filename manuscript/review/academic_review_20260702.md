---
_agent_frontmatter:
  id: "manuscript.review.20260702"
  purpose: "Structured academic review of the manuscript for PLOS ONE submission readiness"
  steward: "manuscript-agent"
  edit_policy: "generated_do_not_edit"
  generator: "academic-manuscript-review skill"
  generated_utc: "2026-07-02T21:00:00Z"
---

# Manuscript Review: From parliamentary claims to parliamentary conduct

## Summary
This manuscript presents a reproducible, multimodal framework for estimating Swedish party policy profiles from motions, speeches, and roll-call votes. The work is methodologically rigorous, with strong transparency practices (deterministic-first pipeline, explicit non-causal boundaries, full provenance tracking). The speech meta-classifier achieves 0.94 accuracy on held-out gold labels. The manuscript is well-structured and close to submission-ready for PLOS ONE.

## Major Issues

### Issue 1: Abstract contains transition text
- **Severity**: Major
- **Location**: Section 01a_abstract, last sentence
- **Problem**: The abstract ends with "The next section states the research question and comparative frame." This is a placeholder/transition that belongs in a section header, not an abstract.
- **Recommendation**: Remove the last sentence. The abstract should stand alone.
- **Expected impact**: Professional abstract that reads as a complete summary.

### Issue 2: Missing generated figures
- **Severity**: Critical
- **Location**: Section 03_results, lines 52-62
- **Problem**: Four figures are referenced but may not exist as generated assets. The figure generation command (`make figures`) timed out when attempted.
- **Recommendation**: Generate figures before submission. Verify each referenced path exists.
- **Expected impact**: Complete manuscript with visual evidence.

### Issue 3: Results section is verbose
- **Severity**: Major
- **Location**: Section 03_results (128 lines)
- **Problem**: The results section contains extensive methodological caveats and interpretation guidance that belongs in the Discussion. For example, lines 70-71, 78, 82-83, 96-98, 118 all restate the non-causal framing already covered in Methodology.
- **Recommendation**: Consolidate repeated caveats. Move detailed interpretation guidance to the Conclusion. Keep Results focused on what was found.
- **Expected impact**: Tighter, more readable results section.

### Issue 4: No explicit hypothesis testing structure
- **Severity**: Major
- **Location**: Section 03_results
- **Problem**: The three hypotheses from the Question section (modality sensitivity, say-do consistency, fulfillment/contradiction diagnostics) are not explicitly addressed with dedicated subsections. The results narrative is thematic but doesn't map clearly to the stated hypotheses.
- **Recommendation**: Add three subsections: "Hypothesis 1: Modality-Sensitive Profiles", "Hypothesis 2: Say-Do Consistency", "Hypothesis 3: Fulfillment and Contradiction Diagnostics". Map existing evidence to each.
- **Expected impact**: Clearer alignment between questions and answers.

### Issue 5: Methodology section missing reproducibility command
- **Severity**: Minor
- **Location**: Section 03_methodology, end
- **Problem**: The methodology mentions reproducibility but doesn't give the exact command to reproduce the full pipeline.
- **Recommendation**: Add a "Reproducibility" subsection with the exact command: `uv run python scripts/update_pipeline.py --cpu-fraction 0.25`
- **Expected impact**: Readers can immediately reproduce results.

## Minor Issues

### Issue 6: Title is too long
- **Location**: Section 01_title
- **Problem**: The title is 18 words. PLOS ONE recommends concise titles.
- **Recommendation**: Shorten to: "Multimodal estimation of Swedish party policy profiles from motions, speeches, and votes"
- **Expected impact**: More readable, better for search.

### Issue 7: Data availability section could be more specific
- **Location**: Section 05_data_availability
- **Problem**: The Zenodo DOI is provided but the repository URL uses a personal GitHub account. For PLOS ONE, a permanent archive is preferred.
- **Recommendation**: Note that the Zenodo DOI is the canonical archival reference. Consider creating a formal release.
- **Expected impact**: Meets PLOS ONE data availability requirements.

### Issue 8: Acknowledgments section uses "the authors" for single author
- **Location**: Section 06_acknowledgments, line 26
- **Problem**: "No external funding was received for this study. The authors received no specific grant..." - inconsistent with single-author declaration.
- **Recommendation**: Change to "The author received no specific grant..."
- **Expected impact**: Consistent voice throughout.

## Journal Fit Assessment
- **Target journal**: PLOS ONE
- **Scope match**: Strong - broad social-science scope, methods-focused, reproducibility emphasis
- **Novelty assessment**: Moderate - novel multimodal combination, but individual components are established
- **Formatting compliance**: Good - Vancouver references, data availability statement, CRediT author contributions all present

## Summary of Changes Made
- [x] Fix abstract transition text
- [x] Add hypothesis subsections to Results
- [x] Add reproducibility command to Methodology
- [x] Fix single-author voice in Acknowledgments
- [x] Shorten title
- [ ] Generate figures (requires `make figures` in manuscript directory)
