# OMR - Optical Music Recognition

A Python application for recognizing musical notes from sheet music images. 

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OrpheusDevTeam/OMR.git
    cd OMR
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 installed. Then, install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the OMR process, execute the `main.py` script with the path to the music sheet images as an argument.

```bash
python3 main.py path/to/your/image.png
```

The script will perform segmentation on the input image. The segmented parts of the music sheet, like staves without lines, will be saved in the `segmented/` directory.

### Environmental variables

The system accepts optional environmental variables:
- `OMR_LOG_DIR`: string path to the directory where the logs are supposed to be stored, overrides default `logs/` directory 

## Pipeline Overview

The Optical Music Recognition (OMR) process is structured into four main stages:

1.  **Preprocessing & Segmentation (`omr/preprocessing/segmenter.py`)**
    *   **Image Binarization**: The input image is first converted to grayscale and then binarized using Otsu's thresholding. This creates a clean black-and-white image, separating the musical symbols from the background.
    *   **Staff Line Detection**: A morphological opening operation is applied with a long, thin horizontal kernel. This effectively isolates the continuous horizontal staff lines, creating a mask that contains only them.
    *   **Staff Line Removal**: The generated staff line mask is used with `cv2.inpaint` to remove the lines from the binarized image. This leaves an image containing only the musical notation (notes, clefs, rests, etc.).
    *   **Staff Segmentation**: The code identifies groups of 5 staff lines to locate each staff system. It then crops the original and the line-free images into separate regions, one for each staff. The `main.py` script saves these line-free staff regions into the `segmented/` directory.

2.  **Symbol Recognition (`NotesRecognitionModule/`) - (Work in Progress)**
    *   **Model**: This stage uses a **YOLO (You Only Look Once) v8** object detection model to find musical symbols within the line-free staff images.
    *   **Training**: The model is being trained on the DeepScores dataset, starting from the pre-trained `yolov8m.pt` weights.
    *   **Output**: The goal of this module is to produce a list of all detected musical symbols (e.g., `notehead-black`, `clef-g`, `rest-quarter`) and their precise bounding box coordinates on the page.

3.  **Postprocessing & Logical Assembly (`omr/postprocessing/combine.py`)**
    *   **Symbol Association**: This is a step that converts the raw, unordered list of detected symbols from the YOLO model into a meaningful musical structure.
    *   **Structure Identification**: It begins by identifying global elements like clefs and time signatures.
    *   **Note Assembly**: It then uses a **KD-Tree** for efficient, proximity-based searching to logically associate note components. For example, it finds the closest `stem` and `flag` for each `notehead` to form a complete note. This process can also identify groups of notes that belong to a beam.
    *   **Output**: The result is a `MusicScore` object, which is a structured Python representation of the sheet music, containing measures, notes, clefs, and other musical information.

4.  **MusicXML Generation (`omr/postprocessing/convert_to_music_xml.py`)**
    *   **Templating**: The final stage takes the structured `MusicScore` object.
    *   **Conversion**: It uses a **Jinja2** template (`musicxml_template.j2`) to render the `MusicScore` data into a standard, well-formed **MusicXML** file. This file can then be opened and edited in any modern music notation software like MuseScore, Sibelius, or our own ORPHEUS Editor.

### Configuration

The `config.yaml` file contains basic configuration for the application, right now supported variables are:
- application name, version and description 
- supported formats of input files

```yaml
app:
  name: orpheus-omr
  version: 0.1.0
  description: Optical Music Recognition tool

formats:
  supported:
    - jpg
    - png
    - pdf
```

## Dependencies

The project relies on the following libraries:

- `pydantic`: For data validation.
- `PyYAML`: For managing configuration.
- `PyMuPDF`, `numpy`, `opencv-python`: For image preprocessing and segmentation.
- `Jinja2`: For templating, likely for generating output formats like MusicXML.