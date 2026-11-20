# Architecture decisions

This file records the decisions that shaped `tabscribe`, why they were made and
what the alternatives were. New decisions are appended in reverse chronological
order.

The guiding principles, in order of priority:

1. **Accuracy first.** Measure, do not guess. Every accuracy number in
   `docs/ACCURACY.md` comes from the harness in `tabscribe-eval`.
2. **Fully offline.** After installation there are zero network calls.
3. **Playable output.** A tab a human cannot play is a failed transcription,
   even if every pitch is correct.
4. **Maintainable.** Small crates, typed interfaces, no hidden state.

---

## D-0001 — Strings are numbered from the highest pitch (string 1) downwards

**Decision.** `Tuning` stores its strings in *string order*: index 0 is string 1,
the top line of an ASCII tab. `Tuning::parse_tuning` accepts a tuning written
in either order and normalises it.

**Why.** The convention of printed tablature is that string 1 is the highest
string. On a ukulele in GCEA tuning the fourth string (G4) sounds *higher* than
the third string (C4), so a pitch-sorted representation would print the tab
upside down. Guitar and bass happen to be monotonic in pitch, which is why the
bug only shows up on the ukulele.

**Alternatives.** Sorting the strings by pitch (simpler, but wrong for the
ukulele) or storing an explicit permutation.

---

## D-0002 — Both ONNX runtimes are optional features; the default build is DSP-only

**Decision.** `tabscribe-core` has two optional cargo features, `neural-tract`
(pure Rust ONNX inference via `tract-onnx`) and `neural-ort` (ONNX Runtime via
`ort`). Neither is enabled by default.

**Why.** Three reasons.

* `ort` has never had a stable release: every published version is a release
  candidate (`2.0.0-rc.15` at the time of writing). Depending on it by default
  would make `tabscribe` a moving target.
* `tract-onnx` is pure Rust, which keeps the binary self-contained, but it is a
  very large dependency: compiling it takes over half an hour on a modest
  machine. Making it the default would make the project unpleasant to build
  and to contribute to.
* The classical DSP engine must work with no model files present anyway, so
  there is no functional reason to force the neural dependency on users who
  only want monophonic transcription.

**Consequences.** `tabscribe song.mp3` works out of the box with no model
files. Polyphonic neural transcription is one flag away:
`cargo build --release --features neural-tract`. The `auto` engine picks the
neural engine when it is compiled in and the material is polyphonic.

**Benchmark.** See `docs/ALGORITHMS.md` for the measured comparison between the
two backends.

---

## D-0003 — The resampler is implemented in-house

**Decision.** Sample rate conversion uses a Kaiser-windowed sinc interpolator
written in this repository (`audio::resample`) rather than `rubato`.

**Why.** The only conversion needed is mono `f32` at an arbitrary ratio. The
implementation is about 120 lines, has better than 80 dB stopband attenuation
and is fully deterministic. `rubato` 5.x is a large dependency whose API
changed substantially in the 5.0 release, and pulling it in for one operation
is not a good trade for a tool whose selling point is that it installs quickly
from source.

**Alternatives.** `rubato` (heavier, better supported), `symphonia`'s built-in
resampler (only integer ratios, not sufficient).

---

## D-0004 — The K-weighting filter follows ITU-R BS.1770-4 exactly

**Decision.** Loudness is measured with the two-stage K-weighting filter and the
gated integration algorithm of BS.1770-4, not with an RMS approximation.

**Why.** The RMS approximation is about 3 dB off for typical material and gets
worse for sparse signals. The standard is short, and implementing it exactly
means the measured LUFS matches what a loudness meter reports.

**Caution.** The denominator coefficients must be normalised exactly once. An
earlier version normalised them twice, which made the filter unstable (the
poles moved outside the unit circle) and produced NaN loudness values. The
unit test `loudness_of_a_full_scale_tone_is_near_minus_three_lufs` catches this.

---

## D-0005 — Tablature assignment is a shortest-path problem over a layered DAG

**Decision.** The assignment optimizer is a Viterbi-style dynamic program over
the states of each chord group. The forward and backward passes are combined to
recover the *k* best alternatives for any single group.

**Why.** The alternative formulations in the literature (greedy left-to-right
assignment, or a beam search) cannot look ahead: a greedy choice at bar 3 can
make bar 40 unplayable. The DP is exact, costs `O(groups × states²)` and the
backward pass gives the alternatives for free.

**Bounded state space.** The number of valid fingerings of a chord grows
combinatorially, so the candidate states of a group are pruned to the most
promising ones (by the local cost) before the DP runs. The limit is
configurable; the default of 256 states per group keeps a four-minute song
well inside a few hundred megabytes.

---

## D-0006 — The JSON output carries alternatives, not just one fingering

**Decision.** The JSON report includes, for every note, the alternative
fingerings that were considered, best first.

**Why.** A tablature is not a uniquely determined object: an ambiguous passage
has several equally valid fingerings, and the performer may prefer another one.
Reporting only the best fingering hides that choice; reporting the alternatives
lets a user (or a GUI) offer the choice.

---

## D-0007 — The v1 product is a CLI; the core library is designed for a GUI

**Decision.** The Android GUI is not part of the v1 milestones. The core
library is the deliverable that a GUI would consume.

**Why.** The transcription pipeline is the hard part and the part with the
accuracy budget. A GUI adds a second large surface (and, on Android, a
JNI/NDK build matrix) without improving the transcription. The core API is
deliberately GUI-friendly: plain data in, plain data out, no file system
access, no global state, and every stage individually callable.

**Later.** The `tabscribe-core` API is the contract for a future GUI: the
pipeline is a sequence of pure functions over `AudioBuffer`, `Transcription` and
`Tab`, so a mobile front end can drive exactly the same code the CLI does.

---

## D-0008 — Quantization is confidence-aware and off by default for the MIDI output

**Decision.** The MIDI file always preserves the performed timing. A separate
quantized MIDI file is only written when `--quantized-midi` is given. The
`--quantize` setting controls the *notated* grid used by the tablature and the
MusicXML output.

**Why.** Over-quantizing is the most common way a transcription stops being
useful: the performed timing carries the feel of the performance. The spec
requires a performance-accurate MIDI file, and a quantized one "optionally
alongside".

---

## D-0009 — The audio pipeline is organised as pure stages over plain data

**Decision.** Every stage of the pipeline is a function from one data type to
another. `AudioBuffer` → `Transcription` → `Tab` → rendered output.

**Why.** It makes every stage testable in isolation (no audio fixtures needed
for the optimizer, no MIDI needed for the renderer), it makes the stages
swappable (the neural and DSP engines implement the same trait), and it keeps
the crate free of I/O so that it can be reused from a GUI.

---

## D-0010 — Models are committed to the repository and verified by SHA-256

**Decision.** The model file lives in `models/` and is tracked by git. Its
SHA-256 is recorded in `models/manifest.json`, and `tabscribe models verify`
checks it.

**Why.** The user asked for the model to be bundled in the repository, and it
keeps a clone fully functional with no extra download step. The checksum makes
a corrupted or substituted file detectable before it reaches the inference
backend, and the manifest records the licence of every model in one reviewable
file.

**Trade-off.** A large binary in git history. The model is a few megabytes, and
the benefit — a clone that works immediately — is worth more than the history
size for a project of this size.

---

## D-0011 — Meter is detected with phase-aware autocorrelation plus accent contrast

**Decision.** The time signature is estimated from the onset envelope: the
tempo and beat phase are found by autocorrelation, the downbeat phase is
refined by maximizing the accent contrast between downbeat and offbeat
positions, and the numerator is chosen from the phase-resolved beat pattern.
Simple modulo-of-bar-period scoring and weighted-profile scoring were both
rejected because they are inconclusive on real material.

**Why.** The reference recording is in 3/4, but unweighted autocorrelation
locks onto its bar period at 46.56 BPM (one third of 140 BPM). A tempo prior
centred at 120 BPM corrects the tempo, but the meter still needs a downbeat
test: the phase that maximizes the difference between downbeat and offbeat
accent strength is the downbeat. The arpeggio fixture moved from 2/4 to 4/4
under this method; that result was accepted because its notes fall one per
beat with the root on beat one.

**Alternatives.** Weighted Krumhansl-style meter profiles (inconclusive),
simple modulo scoring (picks the bar period, not the meter), and a fixed
4/4 assumption (wrong for the reference).

---

## D-0012 — Short notes are removed after onset splitting

**Decision.** After the pipeline splits notes at detected onsets and refines
their starts toward salient onsets, any resulting note shorter than the
cleanup threshold is removed by `cleanup::remove_short_notes`. This cleanup
runs once, after splitting, not iteratively.

**Why.** Onset splitting can create very short tail fragments at note
boundaries (a few milliseconds of residual energy). These fragments are not
musical notes and inflate the note count without improving accuracy. The
reference count fell from 536 to 417 notes after this cleanup was introduced,
and from 417 to 384 with the posterior-HMM engine.

**Alternatives.** Iterative cleanup (removed fragments that were real short
notes), no cleanup (536 notes, too many fragments), and per-engine cleanup
thresholds (inconsistent behaviour across engines).

---

## D-0013 — The tempo prior is a log-normal centred at 120 BPM, one octave wide

**Decision.** Tempo candidates are scored with a log-normal prior centred at 120
BPM and spanning one octave, combined with the autocorrelation peak strength.
This prior is applied when selecting the autocorrelation peak.

**Why.** Unweighted autocorrelation selected the reference's bar period at
46.56 BPM instead of the true 140 BPM. The prior pulls the estimate back
toward a musically reasonable tempo without overriding strong autocorrelation
evidence. The reference result is now 139.7 BPM against the GP's 140 BPM.

**Alternatives.** A flat prior (picks the bar period at 46.56 BPM), a
Gaussian prior centred at 120 BPM (less principled for a scale-invariant
quantity), and no tempo estimation (hard-coded 120 BPM, wrong for fast or
slow material).

---

## D-0014 — Pitch decoding uses a framewise posterior-distribution HMM

**Decision.** Each frame's YIN candidates are interpolated onto semitone bins,
smoothed over five frames, and decoded through a note-state segmenter that
emits one note per sustained pitch state. The YIN period reference is the
first threshold crossing of the difference function, not the global minimum.

**Why.** Random seeded posterior sampling was non-deterministic and hard to
tune. The distribution HMM is deterministic: interpolation spreads a
candidate's probability across neighbouring semitones (so a candidate between
two frets doesn't collapse to the wrong one), five-frame smoothing removes
frame-to-frame jitter, and the segmenter merges frames into musically
meaningful notes. The first threshold crossing (rather than the global
difference-function minimum) avoids octave errors where the global minimum
locks onto a subharmonic.

**Alternatives.** Random seeded sampling (replaced, non-deterministic),
global-minimum YIN reference (70.6% pitch exactness vs 82.0% with first
crossing), and no smoothing (jittery, fragmented notes).

**Consequence.** The posterior-HMM engine produces 384 notes on the reference
against 401 reference events, with 313/401 onsets matched (78.1%) and a mean
onset error of 10.4 ms.

---

## D-0015 — Harmonic salience weights pitch candidates by their FFT partial strengths

**Decision.** Each posterior candidate is weighted by the inverse-harmonic
weighted strength of up to six partials in the frame's FFT magnitude spectrum:
`sum(peak[harmonic × f0] / harmonic)` for harmonic 1..6, raised to
`SALIENCE_EXPONENT` (1.0). Candidates whose frequency explains little of the
frame's energy are suppressed.

**Why.** The difference function alone confuses a fundamental with a harmonic
of another note, producing octave errors. Weighting by harmonic salience
tells a fundamental (strong partials at f0, 2f0, 3f0…) from a harmonic (strong
at only some partials). Exponent 1.0 means the salience breaks ties between
similar-probability candidates without overruling a clearly stronger one.

**Tuning.** Exponents 1.0, 2.0 and 4.0 were compared on the reference: 311,
312 and 313 of 401 onsets matched respectively. The difference is within
noise, so exponent 1.0 (linear) was kept as the simplest choice. Squared
inverse weights (`1/harmonic²`) were also tested and gave no improvement over
linear weights.

**Dominant residual error.** Even with salience, the detected pitches are
predominantly three semitones above the Guitar Pro reference (263/311 matched
pitches at a +3 offset, 84.6% exact). FFT evidence confirms this is a real
property of the recording: at 4.286 s the MP3 fundamental is ≈349.9 Hz (F4,
MIDI 65) where the GP notates D4 (MIDI 62). Lower-register notes are more
complex and are not all explainable by a constant transposition. This is a
recording-vs-notation mismatch, not a detection bug, and is left uncorrected
by the library.

**Alternatives.** No salience weighting (more octave errors), more than six
harmonics (adds Nyquist-limited noise), and exponent ≫ 1 (lets salience
overrule clear probability differences, which worsened results).
