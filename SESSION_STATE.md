# tabscribe Session State — 2026-11-20

## Current Task: Match "Nothing Else Matters" GP5 (filebin.net/vqrdezgzijdd5owm)

### Files
- `/home/user/Untitled.mp3` — Input audio (6.26 MB, "Nothing Else Matters" by Metallica)
- `/home/user/Untitled.gp5` — Expected Guitar Pro 5 file (54.6 KB)
- `/home/user/gp5_expected.json` — All 2680 notes from GP5 (3 tracks: Guitar 1561, Cello 380, Drums 739)
- `/home/user/gp5_guitar_expected.json` — 1561 guitar track notes only

### Best Results (Auto hybrid: polyphonic MF + DSP fallback)

| Metric | Value |
|--------|-------|
| Notes detected | ~3118 |
| Expected notes | 1561 |
| Onset match (50ms) | 79.8% (1245/1561) |
| Pitch correct | 8.8% (110/1245) |
| Tempo | 73.8 BPM (expected: 74 BPM) ✓ |
| Time signature | 3/4 ✓ |

### Other Engine Results

| Engine | Notes | Onset Match | Pitch Correct |
|--------|-------|-------------|---------------|
| DSP (standalone) | 884 | 38.4% | 14.0% |
| Polyphonic (standalone) | 4135 | 80.7% | 11.2% |
| Polyphonic + MF inference | 2812 | 75.7% | 6.9% |
| Onset-gated YIN | 879 | 55.9% | 14.4% |
| Onset-gated HPS | 959 | 59.4% | 8.8% |
| Auto hybrid | 3118 | 79.8% | 8.8% |

### Key Fixes Applied

1. **Tempo detection** (`rhythm.rs`): Lowered prior center 120→90 BPM, added 2:1 tempo correction. Detects 73.8 BPM (matches GP5's 74 BPM).

2. **YIN harmonic locking** (`dsp.rs`): Added sub-harmonic dominance check, missing fundamental check, and HPS fundamental detection.

3. **Polyphonic detector** (`poly.rs`): Added HPS score function, HPS-based fundamental refinement, sub-harmonic consolidation, missing-fundamental pre-pass that infers fundamentals (like E2) from harmonic groups when the fundamental bin is weak.

4. **Onset-gated engine** (`onset_gated.rs`, new): Detects onsets first, then reads frame-level YIN pitch at each onset. Avoids HMM pitch stickiness.

5. **Auto engine** (`mod.rs`): Hybrid polyphonic+DSP — polyphonic for onset coverage, monophonic for accurate single-note pitches.

6. **Pipeline** (`pipeline.rs`): Added leading silence removal to prevent spurious detections during quiet lead-in.

7. **CLI** (`cli.rs`, `commands.rs`): Added OnsetGated engine variant.

### GitHub
- Repo: https://github.com/romankolov/tabscribe
- All changes pushed to main (commit 520b579)
- `.github/workflows/transcribe.yml` — Automated CI that downloads MP3, builds, transcribes, uploads artifacts

### Remaining Challenges to 100% Match

1. **Missing fundamentals**: E2's fundamental (82 Hz) is very weak in the spectrum. The missing-fundamental pre-pass helps but doesn't fully resolve the ambiguity between E2 and its harmonics (B3 = 3rd harmonic).

2. **Polyphonic pitch accuracy**: Even with onset-matched detections, pitch correctness is ~9%. The polyphonic detector often confuses harmonics with fundamentals.

3. **Spurious notes**: ~3100 detected vs 1561 expected. Many are sympathetic resonances and harmonics detected as separate notes.

4. **Leading audio**: ~0.8s of quiet audio before the first note causes some spurious detections (partially mitigated by leading silence removal).
