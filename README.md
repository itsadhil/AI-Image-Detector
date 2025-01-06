# AI Image Detector

This project is a Python-based application that detects whether an image is AI-generated or not. The app uses a pretrained deep learning model and provides a simple graphical user interface (GUI) for easy interaction.

---

## Features

- **User-Friendly Interface**: The app features a GUI built with `tkinter`, making it simple to select and analyze images.
- **Pretrained Model Integration**: Utilizes a Hugging Face pretrained model for classification.
- **Real-Time Results**: Provides confidence scores for each label after analyzing the image.
- **Image Preview**: Displays the selected image for easy verification.

---

## Installation

Follow these steps to set up and run the application:

### Prerequisites

- Python 3.8 or later
- `pip` package manager

### Required Libraries

Install the required Python libraries using the following command:

```bash
pip install torch torchvision transformers pillow
```

---

## Usage

1. Clone or download the repository.
2. Save the provided code as `main.py`.
3. Run the script:
   ```bash
   python main.py
   ```
4. The GUI will open. Click the **Select Image** button to choose an image file.
5. The app will display the image and show the detection results with confidence scores.

---

## How It Works

1. **Model**: The app uses the `microsoft/beit-large-patch16-224-pt22k-ft22k` model by default. This can be replaced with a model specifically trained for AI image detection.
2. **Image Preprocessing**: The selected image is resized and converted to a tensor suitable for the model.
3. **Inference**: The model analyzes the image and outputs confidence scores for each label.
4. **Results Display**: The results are displayed in the GUI, including confidence percentages for each label.

---

## File Structure

```plaintext
.
├── main.py    # Main Python script for the app
├── README.md               # Documentation (this file)
└── requirements.txt        # List of dependencies (optional)
```

---

## Dependencies

The following libraries are used in this project:

- [torch](https://pytorch.org/): For loading and running the pretrained model.
- [transformers](https://huggingface.co/transformers/): To integrate the Hugging Face model.
- [Pillow](https://python-pillow.org/): For image handling.
- [tkinter](https://docs.python.org/3/library/tkinter.html): For building the GUI.

---

## Example Output

- Select an image file using the GUI.
- The app will display the image and provide results like this:

```plaintext
Detection Results:
Label1: 92.45%
Label2: 7.55%
```

---

## Customization

- **Model**: Replace the `MODEL_NAME` variable with a model better suited for AI detection.
- **GUI Design**: Modify the `AIImageDetectorApp` class to adjust the layout or add more features.

---

## Future Improvements

- Integrate a model specifically trained for AI-generated image detection.
- Add support for drag-and-drop image selection.
- Include more detailed feedback about why an image might be classified as AI-generated.

---

## License

This project is open-source and free to use under the MIT License.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pretrained models.
- [PyTorch](https://pytorch.org/) for the machine learning framework.
- The open-source community for resources and tools.



