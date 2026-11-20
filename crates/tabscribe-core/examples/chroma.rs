//! A diagnostic for the chroma and the chord recognition.
//!
//! Run with `cargo run --example chroma -p tabscribe-core`.

use tabscribe_core::transcribe::harmony::{chroma_track, recognise_chord};

const NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// A tone with a few harmonics.
fn tone(freq: f64, seconds: f64, sample_rate: u32) -> Vec<f32> {
    let n = (seconds * sample_rate as f64) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let env = (-t * 2.5).exp();
            let v: f64 = (1..=6)
                .map(|h| (2.0 * std::f64::consts::PI * freq * h as f64 * t).sin() / h as f64)
                .sum();
            (v * env * 0.25) as f32
        })
        .collect()
}

fn main() {
    debug_pipeline();
}

fn debug_pipeline() {
    use tabscribe_core::pipeline::Pipeline;

    let sr = 22050;
    let mut signal = Vec::new();
    for f in [220.0, 261.63, 329.63, 440.0, 329.63, 261.63, 220.0, 196.0] {
        signal.extend(tone(f, 0.35, sr));
    }

    let spectrum = tabscribe_core::transcribe::rhythm::SpectrumConfig::for_sample_rate(sr);
    let envelope = tabscribe_core::transcribe::rhythm::onset_envelope(&signal, sr, &spectrum);
    let est = tabscribe_core::transcribe::rhythm::estimate_tempo(&envelope);
    println!("tempo {:?} phase {:.4}", est.tempo.bpm(), est.phase);
    let beats = tabscribe_core::transcribe::rhythm::track_beats(&envelope, est.tempo);
    println!(
        "beats: {:?}",
        beats
            .iter()
            .map(|b| (b * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );

    let track = chroma_track(&signal, sr);
    println!(
        "chroma frame times: {:?}",
        track
            .times
            .iter()
            .take(6)
            .map(|t| (t * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    for (i, b) in beats.iter().enumerate().take(6) {
        let end = beats.get(i + 1).copied().unwrap_or(b + 0.348);
        let seg = track.between(*b, end);
        let mut s = String::new();
        for (k, v) in seg.iter().enumerate() {
            if *v > 0.05 {
                s.push_str(&format!("{}={:.2} ", NAMES[k], v));
            }
        }
        let (c, _) = recognise_chord(&seg, None).expect("chord");
        println!("beat {i} @ {:.3} -> {} | {}", b, s, c.name());
    }
    let _ = Pipeline::dsp();
}
