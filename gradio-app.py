from transformers import AutoModelForSequenceClassification
import torch
from peft import (
    LoraConfig,
    TaskType,
)

import gradio as gr
from gradio import components as gc
import os
import joblib
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,

)

def main():
    # Define a custom theme with a red primary button
    custom_theme = gr.themes.Glass(primary_hue="blue")

    def predict_with_bert_base(driver_input):
        # Load the LabelEncoder object from the file
        label_encoder_dir = '../label-encoder'
        label_encoder_file_path = os.path.join(label_encoder_dir, 'label_encoder.joblib')
        loaded_le = joblib.load(label_encoder_file_path)
        
        # Load your trained model and tokenizer
        model_path = "../bert-models/best_model"
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        

        
        inputs = tokenizer(driver_input, truncation=True, padding=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted label
        predicted_index = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_index].item()
        original_text_label = loaded_le.inverse_transform([predicted_index])[0]
        return original_text_label

    def predict_with_qlora(driver_input):
        label_encoder_dir = '../label-encoder'
        label_encoder_file_path = os.path.join(label_encoder_dir, 'label_encoder.joblib')
        loaded_le = joblib.load(label_encoder_file_path)
        
        # Specify the path where your model was saved
        qlora_path = "../qlora-models/best_model"
        model_path = "bert-large-uncased"
        
        # Specify the quantization and LoRA configurations as used during training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=5, quantization_config=bnb_config)
        
        # when this loads, it might return the below warning:
        # Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-large-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
        # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

        # this warning is expected since the model is just being loaded. The Lora adapter is being loaded below. The lora adapters have all of the fine-tuned
        # changes which will allow the model to be able to make proper predictions, including the classification layer


        model.load_adapter(qlora_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path) 
        
        inputs = tokenizer(driver_input, truncation=True, padding=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted label
        predicted_index = torch.argmax(outputs.logits, dim=1).item()
        #confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_index].item()
        original_text_label = loaded_le.inverse_transform([predicted_index])[0]
        return original_text_label     

    # Define the function for the Interface
    def driver_prediction(driver_input, model_choice):
        if model_choice=="bert-base-uncased":
            output = predict_with_bert_base(driver_input)
        if model_choice=="bert-large-uncased (loaded with qlora)":
            output = predict_with_qlora(driver_input)
        
        return output


    with gr.Blocks(theme=custom_theme, title="Driver Intent Predictor") as demo:
        gr.Markdown("# Driver Intent Classifier")
        gr.Markdown("This application will classify the driver's intent based upon the models that have been trained.")
        gr.Markdown("Two models are implemented here. The first model was trained using the classic LLM fine-tuning pipeline. The second model was trained using QLoRA.")
        with gr.Row():
            driver_input = gc.Textbox(label="Driver Input", placeholder="Enter the driver's input statement")
        with gr.Row():
            model_choice = gc.Dropdown(label="Choose Trained Model", choices=["bert-base-uncased", "bert-large-uncased (loaded with qlora)"])

        btn = gc.Button("Submit", elem_id="custom_submit_btn")
        output_txt = gc.Textbox(label="Driver Intent")
        btn.click(driver_prediction , inputs=[driver_input, model_choice], outputs=[output_txt])

    demo.launch(server_port=7860,server_name="0.0.0.0")
main()