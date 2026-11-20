//! A diagnostic for the polyphonic detector.

use tabscribe_core::transcribe::poly::{PolyConfig, PolyTranscriber};
use tabscribe_core::transcribe::TranscribeOptions;

fn tone(freq: f64, n: usize, sr: u32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let env = (-t * 2.2).exp();
            let v: f64 = (1..=6)
                .map(|h| (2.0 * std::f64::consts::PI * freq * h as f64 * t).sin() / h as f64)
                .sum();
            (v * env * amp as f64) as f32
        })
        .collect()
}

fn chord(freqs: &[f64], dur: f64, sr: u32) -> Vec<f32> {
    let n = (dur * sr as f64) as usize;
    let mut mixed = vec![0.0f32; n];
    for f in freqs {
        for (i, s) in tone(*f, n, sr, 0.22).into_iter().enumerate() {
            mixed[i] += s;
        }
    }
    mixed
}

fn main() {
    let sr = 22050;
    let mut sig = chord(&[261.63, 329.63, 392.0], 1.9, sr);
    sig.extend(vec![0.0f32; (0.1 * sr as f64) as usize]);
    sig.extend(chord(&[246.94, 293.66, 349.23], 1.9, sr));

    let config = PolyConfig::from_options(sr, &TranscribeOptions::default());
    let frames = PolyTranscriber::new().analyze(&sig, &config);
    for f in frames.iter().step_by(20) {
        let pitches: Vec<String> = f
            .pitches
            .iter()
            .map(|p| format!("{}({:.2})", p.pitch.name(), p.confidence))
            .collect();
        println!("t={:.3} {}", f.time, pitches.join(" "));
    }
}
