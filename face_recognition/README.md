# Face Recognition Using DeepFace Lib
Simple real-time facial detection application with webcam, using:

- **OpenCV** (video capture and drawing)
- **DeepFace** (analysis of emotion, gender, age, and race)

---

## Requirements

- Windows / Linux
- Webcam
- Conda (for environment management)
  - Note 1: You can also use `venv` or `pip` if you prefer, but the instructions here are for Conda.
  - Note 2: Instructions on how to install Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
  - Note 3: GPU acceleration is not required but can improve performance if available.

---

## Installation 
Navigate to the project directory and ensure you have the `environment.yml` file (this file lists the dependencies).

Create and activate the Conda environment:

```bash
conda env create -n <ENV_NAME> -f environment.yml
conda activate <ENV_NAME>
```

Install the project in editable mode (this installs the dependencies from `pyproject.toml`):

```bash
pip install -e .
```

---

## Usage
Run the main script:

```bash
python run.py
```

- A video window will open.
- Press **`q`** to exit.

---

## What the app does

- Detects face in frame with Haar Cascade (OpenCV)
- Every ~2 seconds, analyzes:
  - dominant emotion
  - dominant gender
  - age
  - dominant race
- Displays the results on the detected face(s).

---

## Troubleshooting

### Webcam won't open
Check that the camera is available on the system and is not being used by another app.

### Error importing `cv2` or `deepface`
Confirm that the correct environment is active and reinstall:

```bash
python -m pip install -e .
```

### Slow execution
DeepFace analysis is heavy. This is expected on CPUs, especially without GPU acceleration.