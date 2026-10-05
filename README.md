# VisionPulse — AI Visual Assistant for People with Low Vision

VisionPulse turns an uploaded image into a brief description of the visible
subject and its colors, then offers English browser speech and MP3 playback. It is
an educational assistive tool, not a navigation guarantee or medical device.

## 1. Problem Statement

People with low vision can benefit from a short spoken explanation of an image's
main visible subject and colors. Users should not rely on it as the sole source
for decisions involving physical safety.

## 2. Abstract

The Streamlit app corrects image orientation from EXIF metadata and downsizes each
upload to at most 1024 pixels on its longest side. It identifies each upload by a
SHA-256 hash of the file bytes, so a different image triggers a new analysis even
if the filename is unchanged.

The primary backend is Google's Gemini vision-language model
`gemini-2.5-flash`, accessed through `google-genai`. It is prompted to return a
short English description of the main visible subject, color, and a few directly
observable details. If no `GEMINI_API_KEY` is configured or the API request fails, the
app automatically uses Hugging Face's pretrained
`Salesforce/blip-image-captioning-base` image captioner.

OpenCV and NumPy independently measure mean brightness, contrast, blur using
variance of Laplacian, and dominant colors using k-means. The app shows the
measured color names beside the description. Very dark or blurry images and
low-confidence responses are clearly labeled so the app does not present guesses
as certain facts. The app provides English Web Speech API playback and gTTS MP3
audio.

## 3. Technology Stack

- Python 3.13 (or another version supported by the pinned packages)
- Streamlit for the accessible web UI
- Google Gemini 2.5 Flash through `google-genai` for primary visual description
- Hugging Face Transformers and PyTorch for offline BLIP captioning
- Pillow for image decoding, EXIF correction, and resizing
- OpenCV and NumPy for measured image quality and color analysis
- scikit-learn is pinned as the project ML utility dependency
- English gTTS for MP3 speech; browser Web Speech API for immediate playback
- python-dotenv for local `.env` configuration

## 4. Technical Explanation

### 4.1 Image preprocessing and measurement

Pillow validates image data, applies `ImageOps.exif_transpose`, converts to RGB,
and limits the longest side to 1024 pixels before model inference. OpenCV measures
grayscale mean brightness, standard-deviation contrast, and variance of Laplacian
as a blur indicator. OpenCV k-means clusters sampled RGB pixels; cluster centers
are mapped to simple color names. These are measurable properties, not scene
recognition.

### 4.2 Vision-language model and transfer learning

A vision-language model connects visual input with language output. Gemini accepts both the image and a constrained prompt, returning structured JSON
that the app validates before display. The fallback is Salesforce BLIP, a
pretrained image captioning model loaded once and reused. A pretrained model has
learned visual and language patterns from a large prior training corpus; reusing
those learned representations is an example of transfer learning. The app does
not train the captioner on the user's images.

No finite training or test dataset can contain every object that exists. The
application uses pretrained vision-language models to describe many kinds of
subjects without restricting uploads to a fixed object-label list; coverage and
accuracy are still limited by the models' training and the image quality.

### 4.3 Why the original CNN was replaced

The original custom CNN resized every picture to 32x32 and selected among only ten
CIFAR-10-style classes. It cannot describe arbitrary scenes, multiple objects,
their positions, or visible text. A forced class label can sound confident while
being wrong; turning it into a sentence does not make the result a real caption.
That classifier is no longer in the app's description path. `src/train.py`,
`src/model.py`, and `src/evaluate.py` remain as an explicitly educational baseline
experiment only.

### 4.4 OCR, captions, uncertainty, and limitations

The app requests a concise description, not a complete object inventory or OCR
transcription. BLIP generates a general image caption; it is not a complete object
detector. Neither model is perfectly reliable: vision-language systems can miss
content or hallucinate plausible details. Gemini confidence and measured
darkness/blur add explicit context, but no model can guarantee that every
description is correct or 100% certain.

### 4.5 Accessible voice interface

The full current concise description is passed as JSON-escaped English text to
browser `SpeechSynthesis`, with Listen and Stop controls and a speech-rate slider.
Browser speech prefers an available English voice commonly identified as
feminine; voice availability depends on the user's browser and operating system.
gTTS creates an English MP3 for the current description. The interface uses a `#121212` background,
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

### Q4: Why use a vision-language model instead of listing every possible object?
The set of possible real-world objects is open-ended, so a finite labeled dataset
cannot cover every object. A pretrained vision-language model can describe many
subjects without limiting outputs to a fixed set of training labels, although it
can still make mistakes.

### Q5: How does VisionPulse check image quality?
It calculates pixel-based brightness, contrast, dominant color clusters, and
variance of Laplacian for blur. These values can flag poor image conditions but
cannot establish what objects are present.

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
  reliably read text, enumerate all objects, or assess hazards.
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

Run the representative image-quality, upload-validation, uncertainty, and
English-speech tests:

```powershell
python -m unittest discover -s tests -v
```

Run the real BLIP integration cases as well (downloads/loads the model on first use):

```powershell
$env:VISIONPULSE_RUN_MODEL_TESTS = "1"
python -m unittest discover -s tests -v
```

The legacy CNN training and evaluation files are a baseline experiment, not a
required setup step and not part of the application inference path.
