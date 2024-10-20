import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import ollama
from paddleocr import PaddleOCR

# Load YOLO model and PaddleOCR reader
model = YOLO('yolov10l.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR

def preprocess_image(img):
    """Enhanced preprocessing with noise reduction, histogram equalization, thresholding, and sharpening."""
    if len(img.shape) == 2:  # Convert grayscale to BGR if needed
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (800, 800))  # Resize for consistent input size

    # Apply Gaussian blur for noise reduction
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to grayscale
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better contrast
    adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    sharpened_img = cv2.filter2D(adaptive_thresh, -1, kernel)

    # Convert back to RGB for further processing
    final_img = cv2.cvtColor(sharpened_img, cv2.COLOR_GRAY2BGR)

    # Increase contrast using PIL
    pil_img = Image.fromarray(final_img)
    contrast_img = ImageEnhance.Contrast(pil_img).enhance(2.0)

    return np.array(contrast_img)

# YOLO object detection function
def predict_and_detect(chosen_model, img, conf=0.5, rectangle_thickness=2, text_thickness=1):
    """Run object detection on the image using YOLO."""
    try:
        results = chosen_model.predict(img, conf=conf)
    except Exception as e:
        st.error(f"Error during YOLO prediction: {e}")
        return img, []

    if not results:
        st.warning("No detection results were found.")
        return img, []

    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract box coordinates
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# Function to batch-process detected objects for OCR
def run_ocr_on_boxes(image, boxes):
    """Run OCR on all detected boxes."""
    ocr_results = []
    if not boxes:
        return ocr_results

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_image = image[y1:y2, x1:x2]
        
        if len(cropped_image.shape) == 2:  # Convert grayscale to RGB if needed
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)

        ocr_result = ocr.ocr(cropped_image, cls=True)
        
        if ocr_result and isinstance(ocr_result[0], list):
            extracted_texts = [line[1][0] for line in ocr_result[0]]
            ocr_results.append({'box': (x1, y1, x2, y2), 'ocr_result': extracted_texts})
            st.write(f"OCR Text: {extracted_texts}")

    return ocr_results

# Function to process extracted text with Llama
def call_llama_model(extracted_text):
    """Send extracted text to Llama for drug information extraction."""
    if not extracted_text:
        return "No text found to process."

    prompt = f"""
    Below is text extracted from an OCR related to drugs or tablets.
    Extract drug information like drug name, manufacturing date, expiry date, price, batch number from the text.
    If dates or numbers are found, format them clearly.

    TEXT: {', '.join(extracted_text)}
    """
    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        st.error(f"Error during Llama model call: {e}")
        return "Error processing the text."

# Run OCR on full image as fallback
def run_ocr_on_full_image(image):
    """Run OCR on the full preprocessed image to capture text globally."""
    ocr_result = ocr.ocr(image, cls=True)
    extracted_texts = []
    if ocr_result and isinstance(ocr_result[0], list):
        extracted_texts = [line[1][0] for line in ocr_result[0]]
    return extracted_texts

# Streamlit app layout
st.title("Tablet and Drug Information Extractor")
st.write("Upload an image of a tablet or drug, and the app will perform object detection and OCR.")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    frame = np.array(image)

    # Preprocess the image with enhanced preprocessing
    preprocessed_image = preprocess_image(frame)

    # Object detection and OCR
    result_img, results = predict_and_detect(model, preprocessed_image, conf=0.25)

    extracted_text = []
    if results:
        orig_img = results[0].orig_img if results[0].orig_img is not None else preprocessed_image
        ocr_results = run_ocr_on_boxes(orig_img, results[0].boxes if results[0].boxes else [])
        for ocr_result in ocr_results:
            extracted_text.extend(ocr_result['ocr_result'])

    # If no text found with object detection, run OCR on the entire image
    if not extracted_text:
        extracted_text = run_ocr_on_full_image(preprocessed_image)
        if extracted_text:
            st.write("OCR Text from Full Image:", ', '.join(extracted_text))

    # Display detection and OCR results
    st.image(result_img, caption="Detected Objects", use_column_width=True)

    if extracted_text:
        st.subheader("Extracted Text:")
        st.text(', '.join(extracted_text))

        # Process extracted text with Llama
        llama_response = call_llama_model(extracted_text)
        st.subheader("Processed Drug Information:")
        st.text(llama_response)
    else:
        st.warning("No text extracted from the image.")
