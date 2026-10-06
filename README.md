4# VisionPulse — AI Visual Assistant for People with Low Vision

VisionPulse turns an uploaded image into a cautious, accessible description, then
offers browser speech and downloadable-in-page MP3 playback. It is an educational
assistive tool, not a navigation guarantee or medical device.

## 1. Problem Statement

People with low vision can benefit from a spoken explanation of image contents.
VisionPulse describes uploaded photos, visible text, measured image quality, and
potential visible hazards. It does not label or predict image colors. Users
should not rely on it as the sole
source for decisions involving physical safety.

## 2. Abstract

The Streamlit app corrects image orientation from EXIF metadata and downsizes each
upload to at most 1024 pixels on its longest side. It identifies each upload by a
SHA-256 hash of the file bytes, so a different image triggers a new analysis even
if the filename is unchanged.

The primary backend is Google's Gemini vision-language model
`gemini-2.5-flash`, accessed through `google-genai`. Its JSON response contains a
concise summary, visible objects and positions, visible text, people estimate,
scene notes, and visible safety concerns. If no `GEMINI_API_KEY` is configured or
the API request fails, the app automatically uses Hugging Face's pretrained
`Salesforce/blip-image-captioning-base` image captioner. BLIP is a captioning
fallback: it does not provide reliable OCR, object inventory, people counts, or
hazard analysis, and the app explicitly says so.

OpenCV and NumPy independently measure mean brightness, contrast, and blur using
variance of Laplacian. Very dark or blurry images and low-confidence responses
are accompanied by an uncertainty warning instead of invented detail.
The app provides Web Speech API playback and gTTS MP3 audio.

## 3. Technology Stack

- Python 3.13 (or another version supported by the pinned packages)
- Streamlit for the accessible web UI
- Google Gemini 2.5 Flash through `google-genai` for primary visual description
- Hugging Face Transformers and PyTorch for offline BLIP captioning
- Pillow for image decoding, EXIF correction, and resizing
- OpenCV and NumPy for measured image quality
- scikit-learn is pinned as the project ML utility dependency
- gTTS for MP3 speech; browser Web Speech API for immediate playback
- python-dotenv for local `.env` configuration

## 4. Technical Explanation

### 4.1 Image preprocessing and measurement

Pillow validates image data, applies `ImageOps.exif_transpose`, converts to RGB,
and limits the longest side to 1024 pixels before model inference. OpenCV measures
grayscale mean brightness, standard-deviation contrast, and variance of Laplacian
as a blur indicator. The app intentionally does not infer or label colors.

### 4.2 Vision-language model and transfer learning

A vision-language model connects visual input with language output. Gemini accepts
both the image and a constrained prompt, returning structured JSON that the app
validates before display. The fallback is Salesforce BLIP, a pretrained image
captioning model loaded once and reused. Gemini is prompted to describe the scene,
main subject, position, and surrounding details without inventing unsupported
information or guessing colors. BLIP is allowed a longer generated caption for
additional visible detail. A pretrained model has learned visual and language
patterns from a large prior training corpus; reusing those learned representations
is an example of transfer learning. The app does not train the captioner on the
user's images.

### 4.3 Why the original CNN was replaced

The original custom CNN resized every picture to 32x32 and selected among only ten
CIFAR-10-style classes. It cannot describe arbitrary scenes, multiple objects,
their positions, or visible text. A forced class label can sound confident while
being wrong; turning it into a sentence does not make the result a real caption.
That classifier is no longer in the app's description path. `src/train.py`,
`src/model.py`, and `src/evaluate.py` remain as an explicitly educational baseline
experiment only.

### 4.4 OCR, captions, uncertainty, and limitations

Gemini is prompted to read only visible text (OCR-like interpretation) and to
report uncertainty rather than guess. BLIP generates a general image caption; it
is not presented as OCR or a complete object detector. Neither model is perfectly
reliable: vision-language systems can miss content or hallucinate plausible
details. Gemini confidence and measured darkness/blur add explicit caution, but
the numeric confidence is not a calibrated safety guarantee.

### 4.5 Accessible voice interface

The full current description is passed as JSON-escaped text to browser
`SpeechSynthesis`, with Listen and Stop controls, a speech-rate slider, and
English, Hindi, and Telugu language selection. gTTS creates an MP3 for the current
description and selected language. The interface uses a `#121212` background,
`#00E5FF` actions, `#FFD700` highlights, 20px+ text, 60px+ buttons, and accessible
button labels.

## 5. University Viva Q&A Guide

### Q1: What is a vision-language model?
It is a model that processes visual input and produces or understands language
associated with that input, such as a natural-language image description.

### Q2: What is transfer learning?
It is reusing representations learned from a large prior dataset or task for a
related task, rather than training a model from scratch for every application.
VisionPulse uses pretrained BLIP as its offline captioner.

### Q3: Why did the ten-class CNN fail on arbitrary uploads?
It was trained at 32x32 resolution to classify images into a fixed set of ten
classes. An unrelated image was still forced into one of those labels. It did not
generate language, locate multiple objects, or read text.

### Q4: How do OCR and captioning differ?
OCR extracts readable characters and words from pixels. Image captioning describes
the broader visual scene in language. Gemini is prompted to attempt both; BLIP is
used only for captioning and does not claim to provide OCR.

### Q5: How does VisionPulse check image quality?
It calculates pixel-based brightness, contrast, and variance of Laplacian for
blur. These values can flag poor image conditions but cannot establish what
objects are present. VisionPulse does not provide color labels because automated
color names may be wrong.

### Q6: What is the offline backend?
Salesforce BLIP image-captioning-base, loaded through Transformers and PyTorch. Its
initial weights must be downloaded once; after caching, caption inference can run
without calling Gemini.

### Q7: What is a limitation of vision-language models?
They may omit or hallucinate visual details, misread text, or fail on unusual or
low-quality images. Their outputs are not guaranteed ground truth or safety advice.

### Q8: How is the CNN still used in the project?
Only as a documented baseline experiment for training/evaluation study. It is not
imported by the Streamlit app or used to describe uploads.

## 6. Important Limitations

- Gemini requests transmit the uploaded image to Google's API when a valid API key
  is configured. Follow Google's service terms and avoid uploading sensitive images.
- Without the Gemini API, the local BLIP fallback is a captioner only. It cannot
  reliably perform OCR, enumerate all objects, estimate people, or assess hazards.
- The first BLIP use needs an internet connection to download model weights; later
  runs can use the cached model files.
- Poor lighting, blur, occlusion, unusual scenes, and small text reduce accuracy.
- Color and blur measurements summarize pixels and do not identify the scene.
- No visual model output is a guarantee of safety, navigation support, or diagnosis.

## 7. Privacy

Images are processed in memory by the app and are not saved by VisionPulse.
When `GEMINI_API_KEY` is set, the image is sent to Google's Gemini API for
description. With no key, the app uses the local BLIP model; after the initial
model download, image inference does not require sending images to Gemini. gTTS
also sends description text to Google's text-to-speech service when generating
MP3 audio. Browser speech is synthesized on the user's device.

## 8. Setup & Execution Instructions

### Prerequisites

Python 3.13 is recommended for the pinned dependency set. Create and activate a
virtual environment from the project root:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the pinned packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Gemini, copy `.env.example` to `.env` and add your own key:

```dotenv
GEMINI_API_KEY=your_key_here
```

Do not commit `.env`. Without a key, the first BLIP inference downloads the
caption model, so ensure the machine has internet access for that first run.

Run the application from the project root:

```powershell
streamlit run app.py
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

The legacy CNN training and evaluation files are a baseline experiment, not a
required setup step and not part of the application inference path.
