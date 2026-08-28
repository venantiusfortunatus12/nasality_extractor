# Breton Nasality Acoustic Extractor

A Streamlit application for extracting time-normalised acoustic trajectories from vowel intervals annotated in Praat TextGrids. It is designed for laboratory-phonetic research on Breton nasality, but is language-independent when the annotation contract below is respected.

The application has two modes:

- **Acoustic extraction:** exports vowel-level trajectories of pitch, formants, harmonic amplitudes, nasality-oriented spectral measures, and MFCCs.
- **Optional NAF modelling:** learns a speaker-specific continuous Nasalization from Acoustic Features (NAF) score from labelled oral and nasal calibration fillers.

The exported NAF score is an acoustic, speaker-calibrated estimate of nasality. It is not a direct measure of velum height, velopharyngeal aperture, or nasal airflow.

## Installation

Python 3.10 or newer is recommended.

~~~
cd /Users/alexa/seongwoo_lab/data/nasality/breton_nasality_streamlit_complete
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
~~~

Upload WAV and TextGrid files with identical filename stems.

~~~
speaker01_item001.wav
speaker01_item001.TextGrid
~~~

The default speaker identifier is the part of the filename preceding the first underscore. The separator can be changed in the interface; leaving it empty uses the complete filename stem.

## TextGrid annotation contract

All tiers must be aligned on the same recording time axis. For every vowel, the application retrieves metadata at the vowel midpoint.

| Tier | Required | Purpose |
|---|---|---|
| Segment | Yes | Ordered segment labels, including silence where possible |
| Vowel | Yes | Vowel quality only |
| Word | No | Word/form identifier |
| Stress | No | Stress annotation |
| PhonemicNasality | No | Phonological category: 0 = oral, 1 = nasal |
| NAFTraining | Only for NAF | Calibration filler: 0 = oral, 1 = nasal |

Example:

~~~
Segment
0.00-0.08   k
0.08-0.20   a
0.20-0.29   n
0.29-0.40   t

Vowel
0.08-0.20   a

Word
0.00-0.40   kant

Stress
0.08-0.20   1

PhonemicNasality
0.08-0.20   0

NAFTraining
0.08-0.20   1
~~~

The Vowel tier must contain vowel quality only, for example a, e, ɛ, i, ɔ, or u. Do not encode vowel length, nasalisation, stress, or experimental condition in that label. Store those factors in dedicated tiers.

Accurate vowel boundaries are essential. The application keeps all analysis windows inside the annotated vowel interval, but it cannot correct an inaccurate segmentation.

The comma-separated Nasal segment labels setting, by default m, n, ɲ, ŋ, is used with Segment and PhonemicNasality to derive a phonetic condition for every vowel:

| Derived condition | Definition |
|---|---|
| oral_non_nasal_CVC | PhonemicNasality = 0 and neither adjacent segment is nasal |
| oral_carryover_NVC | PhonemicNasality = 0 and the preceding segment is nasal |
| oral_anticipatory_CVN | PhonemicNasality = 0 and the following segment is nasal |
| oral_coarticulatory_NVN | PhonemicNasality = 0 and both adjacent segments are nasal |
| phonemic_nasal | PhonemicNasality = 1 |

This condition, rather than a simple oral/nasal phonemic split, is used in the preview plots.

## Extracted acoustic measures

The Trajectories sheet contains one row per speaker × token × time point. The default is 31 equally spaced analysis points per vowel.

### Pitch, intensity, and formants

| Measures | Exported columns |
|---|---|
| Fundamental frequency | f0_Hz |
| Intensity and digital level | intensity_dB, digital_RMS, digital_peak_abs |
| Formant frequencies | F1_Hz, F2_Hz, F3_Hz |
| Formant bandwidths | B1_Hz, B2_Hz, B3_Hz |

F0 and intensity are estimated from the original sound rather than from an isolated short spectral slice. Formants are estimated with Praat Burg analysis.

### Harmonics and formant amplitudes

| Measures | Exported columns |
|---|---|
| F0-anchored harmonic frequencies | H01_Hz through H20_Hz |
| F0-anchored harmonic amplitudes | H01_dB through H20_dB |
| Harmonic nearest F1 | A1_Hz, A1_dB, A1_harmonic |
| Harmonic nearest F2 | A2_Hz, A2_dB, A2_harmonic |
| Harmonic nearest F3 | A3_Hz, A3_dB, A3_harmonic |

A1, A2, and A3 are not arbitrary local spectral maxima. They are harmonics anchored to F0 and selected nearest the relevant formant.

### Nasality-oriented spectral measures

| Measure | Exported columns | Interpretation |
|---|---|---|
| Low nasal-pole candidate | P0_Hz, P0_dB, P0_harmonic, P0_prominence_dB | Candidate restricted to H1/H2 in the selected P0 range |
| Higher nasal-pole candidate | P1_Hz, P1_dB, P1_harmonic, P1_prominence_dB | F0-anchored candidate in the selected P1 range |
| First-formant to P0 contrast | A1_P0_dB | Often decreases with nasalisation |
| First-formant to P1 contrast | A1_P1_dB | Secondary cue; often unstable |
| Third-formant to P0 contrast | A3_P0_dB | Spectral-tilt-oriented cue |
| Harmonic tilt | H1_H2_dB | Related to spectral tilt and phonation |
| Broadband tilt | spectral_tilt_dB_per_kHz | Spectral-regression slope over the selected range |
| Spectral moments | spectral_CoG_Hz, spectral_SD_Hz, spectral_skew, spectral_kurtosis | Global spectral distribution |
| Low/high energy | energy_low_nasal, energy_high, nasal_murmur_ratio_dB | Nasal-murmur-oriented energy contrast |

P0 and P1 are candidate measurements, not proof that a physical nasal resonance was recovered. Spectral tilt is also influenced by voice quality, stress, pathology, and recording conditions. No single cue should be interpreted as a universal degree of nasality.

### MFCCs

The default export includes 13 Mel-frequency cepstral coefficients, MFCC_00 through MFCC_12, calculated from 26 Mel filters over 50-8000 Hz. MFCCs are included as multifeature spectral descriptors; they are not intended to be interpreted one by one.

### Validity fields

| Column | Meaning |
|---|---|
| f0_valid | A usable F0 value was obtained |
| formant_valid | F1, F2, and F3 are finite |
| harmonic_valid | A usable A1 harmonic was obtained |
| P0_candidate_valid, P1_candidate_valid | Candidate and prominence could be calculated |
| trajectory_status | Whether a trajectory could be extracted for the token |

## Window placement

The selected analysis window is always constrained to fall entirely inside the vowel interval:

$$
\max\left(e,\frac{w/2}{d}\right)
\leq t_{\mathrm{norm}} \leq
\min\left(1-e,1-\frac{w/2}{d}\right),
$$

where $e$ is edge exclusion, $w$ is the analysis-window duration, and $d$ is vowel duration.

A token shorter than the requested analysis window is retained in the Tokens sheet with trajectory_status equal to vowel_shorter_than_analysis_window, but it has no trajectory rows.

## NAF modelling

NAFTraining provides the acoustic calibration data. It should label only unambiguous filler tokens:

~~~
Clearly oral calibration filler     -> 0
Clearly nasal calibration filler    -> 1
Experimental target                 -> empty
~~~

For instance, non-nasal CVC vowels can be oral fillers and nasal-context NVN vowels can be nasal fillers. The calibration set must be balanced for vowel quality relative to the target data. Otherwise, the model may learn vowel quality rather than nasality.

For each speaker, the app:

1. uses available formant, bandwidth, harmonic, P0/P1, contrast, spectral, energy, and MFCC features;
2. removes features with more than 10% missing values;
3. median-imputes remaining missing values and standardises features;
4. fits PCA to decorrelate the acoustic feature space;
5. fits numeric linear regression to the labelled oral and nasal filler rows;
6. predicts an NAF score for every trajectory row; and
7. reports a token-held-out 75/25 root-mean-square error in NAF_QC.

At least four oral and four nasal labelled tokens are required per speaker. Ten to twenty clean tokens per class, distributed across vowel qualities, are recommended.

PhonemicNasality describes the linguistic category of a token. NAFTraining identifies a reliable acoustic calibration token. They serve different purposes.

## Recommended settings

| Setting | Recommended value |
|---|---:|
| Trajectory points | 31 |
| Analysis window | 30 ms |
| Edge exclusion | 5-15% |
| F0 range | 60-500 Hz |
| Formant ceiling | 5000 Hz for typical lower vocal tracts; 5500 Hz for typical higher vocal tracts |
| P0 candidate range | 200-500 Hz |
| P1 candidate range | 850-1050 Hz |
| Spectral maximum | 5360 Hz |
| Low nasal-energy cutoff | 320 Hz |
| Spectral-tilt range | 500-5000 Hz |
| MFCCs | 13 |
| Mel filters | 26 |

## Output workbook

| Sheet | Contents |
|---|---|
| Tokens | One row per vowel token: metadata, duration, segment context, and eligibility |
| Trajectories | One row per speaker × token × time point, including all acoustic measures and optional NAF score |
| QC_Summary | Proportion of valid F0, formant, harmonic, P0, and P1 measurements |
| NAF_QC | Speaker-wise NAF training counts, selected feature count, PCA dimension, and held-out RMSE |
| Config | Analysis settings used |
| Failures | File pairs that could not be processed |
| Notes | Interpretation and quality-control notes |

The Trajectories sheet is already long/tidy data. Merge it with Tokens by token_id when word, stress, phonemic nasality, or segment context is needed in a downstream model.

## Quality-control workflow

1. Verify vowel boundaries and tier names before extraction.
2. Exclude tokens whose trajectory_status is not ok.
3. Inspect f0_valid, formant_valid, harmonic_valid, P0_candidate_valid, and P1_candidate_valid.
4. Manually inspect a pilot sample of spectra and trajectories.
5. Use A1-P0, F1 bandwidth, and spectral tilt as complementary diagnostics.
6. Do not compare raw A1-P0 or raw NAF values across speakers without within-speaker normalisation or a hierarchical model.
7. Run fPCA or a functional Bayesian model only after acoustic and annotation quality control.

The extractor provides transparent measurements and optional calibrated trajectories. It does not, by itself, establish a phonological contrast; that is the task of the experimental design and statistical model.
