import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your trained model and tokenizer
model_path = "/results/final_qlora_checkpoint"
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")  # Replace with the tokenizer you used during training
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted label
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_index].item()

    # Mapping the index to the respective label
    label_mapping = {
        0: "label_0",
        1: "label_1",
        2: "label_2",
        3: "label_3",
        4: "label_4",
    }
    predicted_label = label_mapping[predicted_index]

    return f"Prediction: {predicted_label}, Confidence: {confidence:.2f}"

iface = gr.Interface(
    fn=predict,                      # The function to be called on user input
    inputs=gr.inputs.Textbox(),      # The input type (a textbox)
    outputs="text"                   # The output type (text)
)

iface.launch()