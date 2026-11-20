# tabscribe Session State — 2026-11-20

## Current Task: Match "Nothing Else Matters" GP5 (filebin.net/vqrdezgzijdd5owm)

### Files
- `/home/user/Untitled.mp3` — Input audio (6.26 MB, "Nothing Else Matters" by Metallica)
- `/home/user/Untitled.gp5` — Expected Guitar Pro 5 file (54.6 KB)
- `/home/user/gp5_expected.json` — All 2680 notes from GP5 (3 tracks: Guitar 1561, Cello 380, Drums 739)
- `/home/user/gp5_guitar_expected.json` — 1561 guitar track notes only

### Best Results (DSP engine)

| Metric | Value |
|--------|-------|
| Notes detected | 884 |
| Expected notes | 1561 |
| Onset match (50ms) | 38.4% (599/1561) |
| Pitch correct | 14.0% (84/599) |
| Tempo | 73.8 BPM (expected: 74 BPM) ✓ |
| Time signature | 3/4 ✓ |

### Key Fixes Applied

1. **Tempo detection** (`rhythm.rs`): Lowered prior center 120→90 BPM, added 2:1 tempo correction. Reference now detects 73.8 BPM (matches GP5's 74 BPM).

2. **YIN harmonic locking** (`dsp.rs`): Added sub-harmonic dominance check — penalizes candidates where f/2 or f/3 has stronger spectral energy. Added missing fundamental check — corrects detected frequency to guitar open string when it's a 2x/3x multiple.

3. **Polyphonic detector** (`poly.rs`): Added sub-harmonic consolidation (promotes f/2/f/3 when its harmonic series is complete), rewrote `notes_from_frames` as onset-aware (only creates notes on pitch transitions, not for continuously-ringing strings), added stricter filtering (confidence ≥ 0.6, duration ≥ 0.1s).

4. **Chord generator** (`gen_synth.py`): Fixed to use correct 3-note triads.

5. **Auto engine** (`mod.rs`): Tempo prior threshold lowered 0.90→0.80.

### GitHub
- Repo: https://github.com/romankolov/tabscribe
- All changes pushed to main
- `.github/workflows/transcribe.yml` — Automated CI that downloads MP3, builds, transcribes, uploads artifacts

### Remaining Challenges to 100% Match

1. **Polyphonic pitch detection**: YIN (monophonic) can only detect one period at a time. When multiple strings ring together (open string arpeggios), it picks the wrong fundamental.

2. **Missing fundamentals**: E2's fundamental (82 Hz) is very weak in the spectrum (mag=2 vs B3's mag=22), making it hard to distinguish from its 3rd harmonic.

3. **Spurious notes**: Polyphonic detector produces 4000+ notes vs 1561 expected. Most are harmonics/sympathetic resonances detected as separate notes.

4. **Leading audio**: ~0.8s of quiet audio before the first note causes spurious detections.

### Next Steps

1. Implement missing fundamental detection using complete harmonic series analysis
2. Add pitch-level onset detection (distinguish newly-plucked vs. ringing strings)
3. Add key/scale-based pitch filtering (prefer notes in the detected key)
4. Consider neural network approach for polyphonic pitch detection
