---
applyTo: "data/**,figures/**,output/**,manuscript/build/**"
---

# Generated Artifact Instructions

- These paths are generated outputs, not hand-edited sources. Prefer regenerating them from the owning scripts instead of editing the files directly.
- If a requested change seems to require modifying one of these files directly, stop and find the producing script or build step first.
- When verifying results, treat timestamps, row counts, and provenance as evidence, but keep manual edits out of generated artifacts.
- For manuscript work, update `manuscript/sections/` and re-render rather than editing `manuscript/build/` files.