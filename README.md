# CHATBOT FOR SUBSTANCE ABUSE SUPPORT 

This project focuses on developing a large language model (LLM)-based chatbot to provide support for substance abuse disorder. The chatbot leverages Intel AI Laptops for its operations and is optimized for CPU inference using Intel® OpenVINO™. The steps included-

•  Model Selection and Fine-tuning: The base model selected for fine-tuning was "NousResearch/Llama-2-7b-chat-hf". The model was fine-tuned using general mental health conversational datasets, which were preprocessed and converted to the ChatML format according to the llama-2-chat-hf input template. The datasets used were:
•	mpingale/mental-health-chat-dataset
•	Amod/mental_health_counseling_conversations
•	heliosbrahma/mental_health_chatbot_dataset
Using the SFTTrainer from the TRL library, I conducted supervised fine-tuning. This involved training the model on the prepared datasets while monitoring performance against evaluation datasets. The trainer incorporated the specified PeftConfig for LoRA-based training in full bit precision(fp 32) and managed sequences up to the defined maximum length (max_seq_length=512).
 
•  Optimization Using Intel® OpenVINO™: The fine-tuned model was converted to the OpenVINO format to optimize inference speed on CPU. The model was quantized to 4 and 8-bit precisions using Intel® OpenVINO™'s quantization tools, ensuring efficient performance without significant loss of accuracy.

•  Integration with Retrieval-Augmented Generation (RAG) System: The vector database for the RAG system contains data specific to substance abuse from the textbook "Substance Abuse Counseling: Theory and Practice" by Patricia Stevens and Robert L. Smith. This integration allows the chatbot to provide more accurate and contextually relevant responses by retrieving and utilizing specific information from the vector database.
•  Deployment and User Interface: The final model and RAG system were deployed using Flask to create a user-friendly web interface.

# Hugging face model repositories:
# fine tuned: https://huggingface.co/HannahJohn/finetuned_llama2-7b-chat-hf/tree/main
# openvino int 8: https://huggingface.co/HannahJohn/openvino-llama2_chat-int4_sym/tree/main
# openvino int4 sys: https://huggingface.co/HannahJohn/openvino-llama2_chat-int4_sym/tree/main

(run app.py to view the chatbot)

