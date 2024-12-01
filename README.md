
# OCR Detection Evaluation Framework

This repository contains tools and scripts for evaluating Optical Character Recognition (OCR) text detection methods. It supports widely used evaluation metrics, including precision, recall, and F-score, along with advanced capabilities to analyze and process detection results.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Files Description](#files-description)
   - [Python Scripts](#python-scripts)
   - [Data and Outputs](#data-and-outputs)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

The repository provides an implementation of the DetEval framework for evaluating OCR text detection tasks. It includes utilities to process ground truth and prediction results in ICDAR and PPOCR formats, compute performance metrics, and visualize evaluation results.

---

## Features

- Implements the DetEval evaluation framework from ICDAR.
- Calculates precision, recall, and F-score for OCR detection tasks.
- Supports one-to-one, one-to-many, and many-to-one matching strategies.
- Converts annotation formats (e.g., PPOCR to ICDAR).
- Utility scripts for editing CSV and text files.

---

## Requirements

- Python 3.7+
- Libraries:
  - `numpy`
  - `json`
  - `zipfile`
  - `csv`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ocr-eval.git
   cd ocr-eval
   ```
2. Install dependencies (if required):
   ```bash
   pip install numpy
   ```

---

## Usage

### Running the Evaluation

1. Ensure the `gt.zip` (ground truth) and `submit.zip` (detection results) files are prepared and formatted according to the ICDAR standard.
2. Run the evaluation script:
   ```bash
   python DetEval.py --g=gt.zip --s=submit.zip --o=./ --p='{"IOU_CONSTRAINT":0.8}'
   ```
3. View the results in the `result.zip` file.

### Format Conversion

Use `DetEval application.py` to convert PPOCR annotations to ICDAR format:
```bash
python DetEval application.py --input PPOCR_annotations.txt --output_dir icdar_format_dir
```

---

## Files Description

### Python Scripts

1. **`DetEval.py`**:
   - Main script for performing evaluation.
   - Supports custom parameters for IOU thresholds and other constraints.

2. **`DetEval application.py`**:
   - Converts PPOCR annotations to ICDAR format.

3. **`edit_csv.py`**:
   - Edits CSV files to include additional metadata.

4. **`rrc_evaluation_funcs.py`**:
   - Utility functions for reading, validating, and processing annotation files.

5. **`test.py`**:
   - Test script for loading and processing annotations.

### Data and Outputs

1. **Input Data**:
   - `gt.zip`: Ground truth annotations in ICDAR format.
   - `submit.zip`: Model predictions.

2. **Output Files**:
   - `result.zip`: Contains evaluation results for each sample.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
