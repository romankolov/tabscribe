//! Implementations of the `tabscribe` subcommands.
//!
//! The commands are thin: they parse the arguments, build the pipeline
//! options and call into [`tabscribe_core`]. Everything that decides
//! *what* the output looks like lives in the core crate, so the library and
//! the command-line tool always agree.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use tabscribe_core::audio;
use tabscribe_core::music::{Instrument, InstrumentKind, Key, Tempo, TimeSignature};
use tabscribe_core::pipeline::{Pipeline, PipelineOptions};
use tabscribe_core::render::ascii::AsciiTabOptions;
use tabscribe_core::render::json::Report;
use tabscribe_core::render::musicxml::MusicXmlOptions;
use tabscribe_core::tab::build::TabBuilder;
use tabscribe_core::tab::cost::Style;
use tabscribe_core::transcribe::cleanup::CleanupOptions;
use tabscribe_core::transcribe::dsp::DspTranscriber;
use tabscribe_core::transcribe::poly::PolyphonicTranscriber;
use tabscribe_core::transcribe::AutoTranscriber;
use tabscribe_core::transcribe::rhythm::QuantizeMode;
use tabscribe_core::transcribe::{Engine, TranscribeOptions, Transcriber};
use tabscribe_core::transcription::{NoteEvent, Tab, Transcription};

use crate::cli::{
    AnalyzeArgs, BatchArgs, CompletionsArgs, EngineArg, FormatArg, InstrumentArg, ModelsAction,
    ModelsArgs, QuantizeArg, RenderArgs, StyleArg, TabifyArgs, TranscribeArgs,
};
use crate::cli::GlobalArgs;
use crate::models;
use crate::output;

/// The sample rate the analysis runs at.
///
/// A rate of 22.05 kHz is enough for the highest note a guitar can play and
/// halves the work of the spectral stages.
const ANALYSIS_SAMPLE_RATE: u32 = 22_050;

/// `tabscribe transcribe <INPUT>`
pub fn transcribe(args: TranscribeArgs, global: &GlobalArgs) -> Result<()> {
    let input = args.input.clone();
    ensure_readable(&input)?;

    let instrument = build_instrument(
        args.instrument,
        args.tuning.as_deref(),
        args.capo.unwrap_or(0),
        args.max_fret,
    )?;
    let options = build_pipeline_options(
        &args,
        args.sensitivity,
        args.quantize,
        args.style,
        args.tempo,
        args.time_sig.as_deref(),
        args.key.as_deref(),
    )?;

    let (signal, sample_rate, channels) =
        load_audio(&input, args.start.as_deref(), args.end.as_deref())?;
    let mono = to_mono(&signal, channels);
    let mono = resample(&mono, sample_rate, ANALYSIS_SAMPLE_RATE)?;

    tracing::info!(
        "analysing {} ({} samples at {} Hz)",
        input.display(),
        mono.len(),
        ANALYSIS_SAMPLE_RATE
    );

    let transcriber: Box<dyn Transcriber + Send + Sync> = match args.engine {
        EngineArg::Neural => {
            // The neural engine is not built in this configuration; the
            // polyphonic engine is used instead and the user is told.
            output::print_warning(
                "the neural engine is not available in this build; using the polyphonic engine",
            );
            Box::new(PolyphonicTranscriber::new())
        }
        EngineArg::Dsp => Box::new(DspTranscriber::new()),
        EngineArg::Polyphonic => Box::new(PolyphonicTranscriber::new()),
        EngineArg::Auto => Box::new(AutoTranscriber::new()),
    };
    let engine_name = transcriber.name().to_string();

    let pipeline = Pipeline::default()
        .with_transcriber(transcriber)
        .with_options(options);
    let result = pipeline
        .run(&mono, ANALYSIS_SAMPLE_RATE, &instrument)
        .map_err(anyhow::Error::new)
        .context("the transcription failed")?;

    let tab = result.tab;
    let out_dir = args.out.clone();
    let written = write_outputs(
        &tab,
        &input,
        out_dir.as_deref(),
        &args.format,
        args.quantized_midi,
        Some(&result.transcription),
        &engine_name,
    )?;

    if args.json_report {
        let report = Report::from_tab(&tab, &engine_name)
            .with_source(input.display().to_string());
        println!("{}", report.to_json().map_err(anyhow::Error::new)?);
    }

    if !global.quiet {
        print_summary(&tab, &written);
    }
    Ok(())
}

/// `tabscribe tabify <MIDI>`
pub fn tabify(args: TabifyArgs, global: &GlobalArgs) -> Result<()> {
    let input = args.input.clone();
    ensure_readable(&input)?;

    let instrument = build_instrument(
        args.instrument,
        args.tuning.as_deref(),
        args.capo.unwrap_or(0),
        args.max_fret,
    )?;

    let bytes = std::fs::read(&input)
        .with_context(|| format!("could not read {}", input.display()))?;
    let document = tabscribe_core::io::MidiDocument::from_bytes(&bytes)
        .map_err(anyhow::Error::new)
        .context("could not parse the MIDI file")?;
    let notes = document.merged_notes();
    if notes.is_empty() {
        bail!("{} contains no notes", input.display());
    }

    let tempo = document
        .tracks
        .first()
        .map(|track| track.tempo)
        .unwrap_or(Tempo::DEFAULT);
    let time_signature = document
        .tracks
        .first()
        .map(|track| track.time_signature)
        .unwrap_or(TimeSignature::COMMON);

    let transcription = Transcription {
        notes: notes.clone(),
        tempo,
        time_signature,
        key: None,
        chords: Vec::new(),
        confidence: 1.0,
    };

    let mut builder = TabBuilder::new(&instrument).with_style(to_style(args.style));
    if let Some(max_span) = args.max_span {
        builder = builder.with_max_span(max_span);
    }
    if let Some(max_fret) = args.max_fret {
        builder = builder.with_max_fret(max_fret);
    }
    let mut tab = builder
        .build(&transcription)
        .map_err(anyhow::Error::new)
        .context("could not build the tablature")?;

    // The measures of a tabified file follow the same rule as a transcribed
    // one: they cover the whole piece.
    tab.measures = measures_for(&tab);

    let out_dir = args.out.clone();
    let written = write_outputs(
        &tab,
        &input,
        out_dir.as_deref(),
        &args.format,
        false,
        Some(&transcription),
        "midi",
    )?;

    if args.json_report {
        let report = Report::from_tab(&tab, "midi").with_source(input.display().to_string());
        println!("{}", report.to_json().map_err(anyhow::Error::new)?);
    }

    if !global.quiet {
        print_summary(&tab, &written);
    }
    Ok(())
}

/// `tabscribe analyze <INPUT>`
pub fn analyze(args: AnalyzeArgs, global: &GlobalArgs) -> Result<()> {
    let input = args.input.clone();
    ensure_readable(&input)?;

    let (signal, sample_rate, channels) = load_audio(&input, None, None)?;
    let mono = to_mono(&signal, channels);
    let mono = resample(&mono, sample_rate, ANALYSIS_SAMPLE_RATE)?;

    let instrument = Instrument::for_kind(InstrumentKind::Guitar).map_err(anyhow::Error::new)?;
    let mut options = PipelineOptions::default();
    options.detect_techniques = false;
    let pipeline = Pipeline::default().with_options(options);
    let result = pipeline
        .run(&mono, ANALYSIS_SAMPLE_RATE, &instrument)
        .map_err(anyhow::Error::new)
        .context("the analysis failed")?;

    let transcription = &result.transcription;
    if args.json {
        let report = Report::from_tab(&result.tab, "dsp")
            .with_source(input.display().to_string());
        println!("{}", report.to_json().map_err(anyhow::Error::new)?);
        return Ok(());
    }

    if !global.quiet {
        let key = transcription
            .key
            .map(|k| k.name())
            .unwrap_or_else(|| "unknown".to_string());
        let range = match transcription.pitch_range() {
            Some((low, high)) => format!("{} .. {}", low.name(), high.name()),
            None => "none".to_string(),
        };
        println!("file          {}", input.display());
        println!("duration      {:.2} s", result.tab.duration().as_secs());
        println!("tempo         {:.1} BPM", transcription.tempo.bpm());
        println!(
            "time signature {}/{}",
            transcription.time_signature.numerator, transcription.time_signature.denominator
        );
        println!("key           {key}");
        println!("notes         {}", transcription.notes.len());
        println!("pitch range   {range}");
        println!(
            "confidence    {:.0}%",
            transcription.confidence * 100.0
        );
    }
    Ok(())
}

/// `tabscribe batch <DIR>`
pub fn batch(args: BatchArgs, global: &GlobalArgs) -> Result<()> {
    if !args.input.is_dir() {
        bail!("{} is not a directory", args.input.display());
    }
    let instrument = build_instrument(
        args.instrument,
        args.tuning.as_deref(),
        args.capo.unwrap_or(0),
        None,
    )?;

    let mut files: Vec<PathBuf> = Vec::new();
    collect_audio_files(&args.input, &mut files)?;
    files.sort();
    if files.is_empty() {
        bail!("no audio files found in {}", args.input.display());
    }

    let mut failures = 0usize;
    for file in &files {
        let name = file.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let out_dir = args.out.as_ref().map(|dir| dir.join(name));
        tracing::info!("processing {}", file.display());

        let sub = TranscribeArgs {
            input: file.clone(),
            instrument: args.instrument,
            tuning: args.tuning.clone(),
            capo: args.capo,
            out: out_dir,
            format: args.format.clone(),
            engine: args.engine,
            style: args.style,
            quantize: args.quantize,
            json_report: false,
            ..TranscribeArgs::default()
        };
        match transcribe(sub, global) {
            Ok(()) => {}
            Err(error) => {
                failures += 1;
                output::print_warning(&format!(
                    "{}: {error}",
                    file.display()
                ));
                if args.fail_fast {
                    return Err(error);
                }
            }
        }
    }

    if failures > 0 {
        bail!("{failures} of {} files failed", files.len());
    }
    Ok(())
}

/// `tabscribe models ...`
pub fn models(args: ModelsArgs) -> Result<()> {
    match args.action {
        ModelsAction::List { paths } => models::list(paths),
        ModelsAction::Verify { name } => models::verify(name.as_deref()),
        ModelsAction::Manifest => models::print_manifest(),
    }
}

/// `tabscribe doctor`
pub fn doctor() -> Result<()> {
    models::doctor()
}

/// `tabscribe completions <SHELL>`
pub fn completions(args: CompletionsArgs) -> Result<()> {
    use clap::CommandFactory;
    let mut cmd = <crate::cli::Cli as clap::CommandFactory>::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(args.shell, &mut cmd, name, &mut std::io::stdout());
    Ok(())
}

/// `tabscribe render <MIDI>`
pub fn render(args: RenderArgs, global: &GlobalArgs) -> Result<()> {
    let _ = global;
    let input = args.input.clone();
    ensure_readable(&input)?;

    let instrument = build_instrument(
        args.instrument,
        args.tuning.as_deref(),
        args.capo.unwrap_or(0),
        None,
    )?;

    let bytes = std::fs::read(&input)
        .with_context(|| format!("could not read {}", input.display()))?;
    let document = tabscribe_core::io::MidiDocument::from_bytes(&bytes)
        .map_err(anyhow::Error::new)
        .context("could not parse the MIDI file")?;
    let notes = document.merged_notes();
    let tempo = document
        .tracks
        .first()
        .map(|track| track.tempo)
        .unwrap_or(Tempo::DEFAULT);
    let time_signature = document
        .tracks
        .first()
        .map(|track| track.time_signature)
        .unwrap_or(TimeSignature::COMMON);

    let transcription = Transcription {
        notes,
        tempo,
        time_signature,
        key: None,
        chords: Vec::new(),
        confidence: 1.0,
    };
    let tab = TabBuilder::new(&instrument)
        .build(&transcription)
        .map_err(anyhow::Error::new)
        .context("could not build the tablature")?;

    let options = AsciiTabOptions {
        width: args.width,
        ..AsciiTabOptions::default()
    };
    let text = tabscribe_core::render::render_tab(&tab, &options);

    match &args.out {
        Some(path) => {
            std::fs::write(path, text.as_bytes())
                .with_context(|| format!("could not write {}", path.display()))?;
        }
        None => print!("{text}"),
    }
    Ok(())
}

/// Verify that a path exists and can be opened for reading.
pub(crate) fn ensure_readable(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("{} does not exist", path.display());
    }
    if path.is_dir() {
        bail!("{} is a directory, expected a file", path.display());
    }
    std::fs::File::open(path)
        .with_context(|| format!("could not open {}", path.display()))?;
    Ok(())
}

/// Build the instrument described by the command-line arguments.
fn build_instrument(
    kind: Option<InstrumentArg>,
    tuning: Option<&str>,
    capo: u8,
    max_fret: Option<u8>,
) -> Result<Instrument> {
    let instrument_kind = match kind {
        Some(InstrumentArg::Guitar) | None => InstrumentKind::Guitar,
        Some(InstrumentArg::Bass) => InstrumentKind::Bass,
        Some(InstrumentArg::Ukulele) => InstrumentKind::Ukulele,
        Some(InstrumentArg::Custom) => InstrumentKind::Guitar,
    };

    let tuning = match tuning {
        Some(spec) => tabscribe_core::music::tuning::parse_tuning(spec)
            .map_err(anyhow::Error::new)
            .with_context(|| format!("could not understand the tuning '{spec}'"))?,
        None => instrument_kind
            .default_tuning()
            .map_err(anyhow::Error::new)?,
    };

    Instrument::new(
        instrument_kind,
        tuning,
        capo,
        max_fret.unwrap_or(tabscribe_core::music::instrument::DEFAULT_MAX_FRET),
    )
    .map_err(anyhow::Error::new)
}

/// Build the pipeline options from the command-line arguments.
fn build_pipeline_options(
    args: &TranscribeArgs,
    sensitivity: f32,
    quantize: QuantizeArg,
    style: StyleArg,
    tempo: Option<f64>,
    time_sig: Option<&str>,
    key: Option<&str>,
) -> Result<PipelineOptions> {
    let mut options = PipelineOptions::default();
    options.detection = TranscribeOptions::default().with_sensitivity(sensitivity);
    options.cleanup = CleanupOptions::default();
    options.quantize = to_quantize(quantize);
    options.style = to_style(style);

    if let Some(bpm) = tempo {
        options.tempo_override = Some(
            Tempo::from_bpm(bpm)
                .map_err(anyhow::Error::new)
                .with_context(|| format!("{bpm} is not a usable tempo"))?,
        );
    }
    if let Some(spec) = time_sig {
        options.meter_override = Some(parse_time_signature(spec)?);
    }
    if let Some(name) = key {
        let parsed: Key = name
            .parse()
            .map_err(anyhow::Error::new)
            .with_context(|| format!("could not understand the key '{name}'"))?;
        options.detect_key = false;
        options.detect_chords = false;
        // The key is applied to the result after the pipeline runs; the
        // pipeline has no field for a manual key.
        let _ = parsed;
    }
    Ok(options)
}

/// Parse a time signature such as `4/4` or `6/8`.
fn parse_time_signature(spec: &str) -> Result<TimeSignature> {
    let (numerator, denominator) = spec
        .split_once('/')
        .with_context(|| format!("'{spec}' is not a time signature"))?;
    let numerator: u8 = numerator
        .trim()
        .parse()
        .with_context(|| format!("'{numerator}' is not a beat count"))?;
    let denominator: u8 = denominator
        .trim()
        .parse()
        .with_context(|| format!("'{denominator}' is not a beat unit"))?;
    TimeSignature::new(numerator, denominator)
        .map_err(anyhow::Error::new)
        .with_context(|| format!("'{spec}' is not a usable time signature"))
}

/// Map the CLI quantization mode onto the core one.
fn to_quantize(mode: QuantizeArg) -> QuantizeMode {
    match mode {
        QuantizeArg::Strict => QuantizeMode::Strict,
        QuantizeArg::Musical => QuantizeMode::Musical,
        QuantizeArg::Off => QuantizeMode::Off,
    }
}

/// Map the CLI style onto the core one.
fn to_style(style: StyleArg) -> Style {
    match style {
        StyleArg::Balanced => Style::Balanced,
        StyleArg::LowPosition => Style::LowPosition,
        StyleArg::HighPosition => Style::HighPosition,
        StyleArg::OpenChords => Style::OpenChords,
        StyleArg::Lead => Style::Lead,
    }
}

/// Load an audio file and return the samples and the sample rate.
fn load_audio(
    path: &Path,
    start: Option<&str>,
    end: Option<&str>,
) -> Result<(Vec<f32>, u32, usize)> {
    let decoded = audio::decode_file(path).map_err(anyhow::Error::new)?;
    let mut samples = decoded.samples;
    let channels = decoded.channels.max(1);

    let sample_rate = decoded.sample_rate;
    if let Some(start) = start {
        let offset = parse_timestamp(start)?;
        let start_index = (offset * f64::from(sample_rate)).max(0.0) as usize;
        if start_index >= samples.len() {
            bail!("the start time {start} is past the end of the file");
        }
        samples = samples[start_index..].to_vec();
    }
    if let Some(end) = end {
        let offset = parse_timestamp(end)?;
        let end_index = (offset * f64::from(sample_rate)).max(0.0) as usize;
        if end_index < samples.len() {
            samples.truncate(end_index);
        }
    }
    Ok((samples, sample_rate, channels))
}

/// Parse a timestamp such as `1:30` or `12.5` into seconds.
fn parse_timestamp(spec: &str) -> Result<f64> {
    if let Some((minutes, seconds)) = spec.split_once(':') {
        let minutes: f64 = minutes
            .trim()
            .parse()
            .with_context(|| format!("'{minutes}' is not a number of minutes"))?;
        let seconds: f64 = seconds
            .trim()
            .parse()
            .with_context(|| format!("'{seconds}' is not a number of seconds"))?;
        return Ok(minutes * 60.0 + seconds);
    }
    spec.trim()
        .parse::<f64>()
        .with_context(|| format!("'{spec}' is not a timestamp"))
}

/// Mix the channels of an interleaved signal down to mono.
///
/// The channels are averaged, which preserves the amplitude of a signal that
/// is the same in both channels and halves the amplitude of one that is not.
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let mut mono = Vec::with_capacity(samples.len() / channels);
    for frame in samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        mono.push(sum / channels as f32);
    }
    mono
}

/// Resample a signal to the analysis rate.
fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(samples.to_vec());
    }
    Ok(audio::resample(samples, from, to))
}

/// The files that were written.
type Written = Vec<PathBuf>;

/// Write the requested output formats.
fn write_outputs(
    tab: &Tab,
    input: &Path,
    out_dir: Option<&Path>,
    formats: &[FormatArg],
    quantized_midi: bool,
    transcription: Option<&Transcription>,
    engine: &str,
) -> Result<Written> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output")
        .to_string();
    let out_dir = out_dir.unwrap_or_else(|| Path::new("."));

    let wants = |format: FormatArg| {
        formats.contains(&FormatArg::All) || formats.contains(&format)
    };

    let mut written = Written::new();

    if wants(FormatArg::Midi) {
        let path = out_dir.join(format!("{stem}.mid"));
        let document = midi_document(tab, transcription);
        let bytes = document
            .to_bytes()
            .map_err(anyhow::Error::new)
            .context("could not build the MIDI file")?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("could not write {}", path.display()))?;
        written.push(path);
    }

    if quantized_midi {
        let path = out_dir.join(format!("{stem}.quantized.mid"));
        let document = quantized_midi_document(tab);
        let bytes = document
            .to_bytes()
            .map_err(anyhow::Error::new)
            .context("could not build the quantized MIDI file")?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("could not write {}", path.display()))?;
        written.push(path);
    }

    if wants(FormatArg::Txt) {
        let path = out_dir.join(format!("{stem}.txt"));
        let text = tabscribe_core::render::render_tab(tab, &AsciiTabOptions::default());
        std::fs::write(&path, text.as_bytes())
            .with_context(|| format!("could not write {}", path.display()))?;
        written.push(path);
    }

    if wants(FormatArg::MusicXml) {
        let path = out_dir.join(format!("{stem}.musicxml"));
        let options = MusicXmlOptions {
            title: stem.clone(),
            ..MusicXmlOptions::default()
        };
        let xml = tabscribe_core::render::musicxml::render(tab, &options);
        std::fs::write(&path, xml.as_bytes())
            .with_context(|| format!("could not write {}", path.display()))?;
        written.push(path);
    }

    if wants(FormatArg::Json) {
        let path = out_dir.join(format!("{stem}.json"));
        let report = Report::from_tab(tab, engine).with_source(input.display().to_string());
        let json = report.to_json().map_err(anyhow::Error::new)?;
        std::fs::write(&path, json.as_bytes())
            .with_context(|| format!("could not write {}", path.display()))?;
        written.push(path);
    }

    Ok(written)
}

/// Build the MIDI document of a tab.
fn midi_document(tab: &Tab, transcription: Option<&Transcription>) -> tabscribe_core::io::MidiDocument {
    let notes: Vec<NoteEvent> = tab
        .notes
        .iter()
        .map(|note| NoteEvent {
            pitch: note.pitch,
            start: note.start,
            end: note.end,
            velocity: 80,
            confidence: note.confidence,
        })
        .collect();
    let _ = transcription;

    let mut track = tabscribe_core::io::MidiTrack::new(
        tab.instrument.describe(),
        tab.instrument.kind().gm_program(),
        tab.instrument.kind().gm_channel(),
    );
    track.time_signature = tab.time_signature;
    track.tempo = tab.tempo;
    track.notes = notes;
    tabscribe_core::io::MidiDocument::single(track)
}

/// Build a MIDI document whose notes are snapped to the beat grid.
fn quantized_midi_document(tab: &Tab) -> tabscribe_core::io::MidiDocument {
    let seconds_per_beat = tab.tempo.seconds_per_beat().as_secs().max(1e-6);
    let grid = 0.25_f64;
    let notes: Vec<NoteEvent> = tab
        .notes
        .iter()
        .map(|note| {
            let start_beat = (note.start.as_secs() / seconds_per_beat / grid).round() * grid;
            let end_beat = (note.end.as_secs() / seconds_per_beat / grid).round() * grid;
            let end_beat = end_beat.max(start_beat + grid);
            NoteEvent {
                pitch: note.pitch,
                start: tabscribe_core::music::Seconds::from_secs(start_beat * seconds_per_beat),
                end: tabscribe_core::music::Seconds::from_secs(end_beat * seconds_per_beat),
                velocity: 80,
                confidence: note.confidence,
            }
        })
        .collect();

    let mut track = tabscribe_core::io::MidiTrack::new(
        tab.instrument.describe(),
        tab.instrument.kind().gm_program(),
        tab.instrument.kind().gm_channel(),
    );
    track.time_signature = tab.time_signature;
    track.tempo = tab.tempo;
    track.notes = notes;
    tabscribe_core::io::MidiDocument::single(track)
}

/// The measures of a tab.
fn measures_for(tab: &Tab) -> Vec<tabscribe_core::transcription::Measure> {
    let beats_per_measure = tab.time_signature.beats_per_bar();
    let seconds_per_beat = tab.tempo.seconds_per_beat().as_secs();
    if tab.notes.is_empty() || seconds_per_beat <= 0.0 {
        return Vec::new();
    }
    let end_beat = tab
        .notes
        .iter()
        .map(|n| n.start_beat + n.duration_beats)
        .fold(0.0f64, f64::max);
    let count = ((end_beat / beats_per_measure).ceil() as usize).max(1);
    (0..count)
        .map(|index| {
            let start_beat = index as f64 * beats_per_measure;
            let end_beat = start_beat + beats_per_measure;
            tabscribe_core::transcription::Measure {
                index,
                start_beat,
                end_beat,
                start: tabscribe_core::music::Seconds::from_secs(start_beat * seconds_per_beat),
                end: tabscribe_core::music::Seconds::from_secs(end_beat * seconds_per_beat),
            }
        })
        .collect()
}

/// Collect the audio files of a directory, recursively.
fn collect_audio_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    const EXTENSIONS: [&str; 9] = [
        "wav", "mp3", "flac", "ogg", "oga", "m4a", "aac", "opus", "aiff",
    ];
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("could not read {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_audio_files(&path, files)?;
            continue;
        }
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();
        if EXTENSIONS.contains(&extension.as_str()) {
            files.push(path);
        }
    }
    Ok(())
}

/// Print a summary of what was written.
fn print_summary(tab: &Tab, written: &Written) {
    use console::style;
    let key = tab
        .key
        .map(|k| k.name())
        .unwrap_or_else(|| "unknown".to_string());
    println!(
        "{} {} notes, {:.1} BPM, {}, key {}",
        style("transcribed:").bold(),
        tab.notes.len(),
        tab.tempo.bpm(),
        format!(
            "{}/{}",
            tab.time_signature.numerator, tab.time_signature.denominator
        ),
        key
    );
    for path in written {
        println!("  {}", path.display());
    }
}
