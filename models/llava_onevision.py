import json
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from models.base_vlm import Base_VLM_Model

config = json.load(open("./config.json"))
inf_prompts = json.load(open("./models/inf_prompts.json"))

INF_SCALING_TEMP = "{user_prompt}\n\n{summary}\n\n{caption}\n\n{answer}"
PRM_INF_SCALING_TEMP = "{caption}\n\n{user_prompt}"

# build bard class
class Llava_Onevision_Model(Base_VLM_Model):
    def __init__(self, model_id, device="cuda:0", max_new_tokens=30, use_inf_scaling=False, use_prm_mcts=False, scale_at_inf=False):
        super().__init__(model_id, device, max_new_tokens, use_inf_scaling, use_prm_mcts, scale_at_inf)

    def get_model(self, model_id):
        # model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            cache_dir=config["model_cache"]).to(self.device)

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=config["model_cache"])
        return model, processor
    
    def generate_response(self, encoded_image, user_prompt, do_sample=True, temperature=1.0, top_p=0.95):
        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=encoded_image, text=prompt, return_tensors='pt').to(self.device, torch.bfloat16)
        output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=do_sample, temperature=temperature, top_p=top_p)
        response = self.processor.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        return response
