from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_name='HannahJohn/openvino-llama2_chat-int4_sym'

ov_model = OVModelForCausalLM.from_pretrained(model_name,device='CPU')
tokenizer = AutoTokenizer.from_pretrained(model_name)
