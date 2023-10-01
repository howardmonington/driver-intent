# Driver Intent Classifier with QLoRA

#### -- Project Status: Completed

## Project Intro/Objective
In this project, I created my own dataset for driver-intent classification. I then used this dataset to compare the fine-tuning pipelines for the traditional vs. the QLoRA techniques. Finally, I created an application that is capable of loading either of the models and making a prediction on a driver's intent based upon a text input. The project is fully dockerized so that it can be easily replicated on any computer. For a fully annotated explanation of each pipeline, please see the notebooks.

### Methods Used
* Traditional Fine-Tuning Pipeline
* QLoRA


### Technologies
* Python
* Pandas, jupyter
* Pytorch
* Docker
* Transformers
* Callbacks

## Project Description
In the traditional method, fine-tuning a large AI model required hefty computational power and expensive GPU resources, making it inaccessible to the open-source community. The model would be fine tuned on a larger data type, and then it was necessary to undertake a process known as 4-bit quantization to run the model on a consumer-grade GPU post-fine-tuning . This process optimized the model to use fewer resources but sacrificed the full power of the model, diminishing the overall results.

QLoRA addresses this predicament, offering a win-win scenario. This optimized method allows for fine-tuning of large LLMs using just a single GPU while maintaining the high performance of a full 16-bit model in 4-bit quantization. With QLoRA, the barrier to entry for fine-tuning larger, more sophisticated models has been significantly lowered. This broadens the scope of projects that the AI community can undertake, fostering innovation and facilitating the creation of more efficient and powerful applications.

In this project, I created notebooks and Python scripts to compare the traditional method of fine-tuning with the QLoRA method. I fine-tuned both models on a synthetic dataset for driver-intent recognition. I created this dataset myself. It consists of 6 classes that each contain 200 samples. It is a balanced classification dataset. Finally, I created a Gradio application so that I can demonstrate how to load and utilize these models in the real-world. This project might be useful for car companies because an LLM can be implemented in the on-board computer in order to detect the driver's intent. The driver can simply speak whatever they want, and the car would be able to recognize the intent and perform an action.  


## Featured Notebooks/Analysis/Deliverables
* BERT Traditional Pipeline: bert.ipynb
* QLoRA Pipeline: qlora.ipynb
* Gradio Application: gradio-app.py
* [QLoRA Youtube Video on my YouTube Channel](https://www.youtube.com/watch?v=n90tKMDQUaY&t=1s&ab_channel=LukeMonington)
* [QLoRA Tutorial on my YouTube Channel](https://www.youtube.com/watch?v=2bkrL2ZcOiM&ab_channel=LukeMonington)
* [Medium Article that I wrote on QLoRA](https://medium.com/@lukemoningtonAI/fine-tuning-llms-in-4-bit-with-qlora-2982cddcd459)


## Contributing Members

**[Luke Monington](https://github.com/lukemonington)**

## Contact
* I can be reached at lukemonington@aol.com.
