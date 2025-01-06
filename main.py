import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Load the pretrained model
MODEL_NAME = "microsoft/beit-large-patch16-224-pt22k-ft22k"  # Replace with a model suited for AI detection
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

def detect_ai_image(image_path):
    """
    Detect if an image is AI-generated using a pretrained model.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        dict: Classification results with labels and confidence scores.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Perform inference
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get classification results
        labels = model.config.id2label
        results = {labels[i]: float(probabilities[0][i]) for i in range(len(labels))}
        return results
    except Exception as e:
        return {"error": str(e)}

# GUI application
class AIImageDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Detector")
        self.root.geometry("600x400")

        # UI elements
        self.label = tk.Label(root, text="Select an image to check if it is AI-generated", font=("Arial", 14))
        self.label.pack(pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.button = tk.Button(root, text="Select Image", command=self.load_image, font=("Arial", 12), bg="blue", fg="white")
        self.button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not file_path:
            return

        # Display the image
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        img = ImageTk.PhotoImage(image)
        self.image_label.config(image=img)
        self.image_label.image = img

        # Perform AI detection
        self.detect_image(file_path)

    def detect_image(self, image_path):
        self.result_label.config(text="Processing...")
        results = detect_ai_image(image_path)
        if "error" in results:
            messagebox.showerror("Error", results["error"])
            self.result_label.config(text="")
        else:
            result_text = "\n".join([f"{label}: {confidence:.2%}" for label, confidence in results.items()])
            self.result_label.config(text=f"Detection Results:\n{result_text}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AIImageDetectorApp(root)
    root.mainloop()
