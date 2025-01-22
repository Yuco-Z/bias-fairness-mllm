import json
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from models.base_vlm import Base_VLM_Model

config = json.load(open("./config.json"))
inf_prompts = json.load(open("./models/inf_prompts.json"))

INF_SCALING_TEMP = "{user_prompt}\n\n{summary}\n\n{caption}\n\n{answer}"
PRM_INF_SCALING_TEMP = "{caption}\n\n{user_prompt}"

# build bard class
class Llama32_Model(Base_VLM_Model):
    def __init__(self, model_id, device="cuda:0", max_new_tokens=30, use_inf_scaling=False, use_prm_mcts=False, scale_at_inf=False):
        super().__init__(model_id, device, max_new_tokens, use_inf_scaling, use_prm_mcts, scale_at_inf)

    def get_model(self, model_id):
        # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=config["model_cache"],
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=config["model_cache"])
        return model, processor
    
    def generate_response(self, encoded_image, user_prompt, do_sample=True, temperature=1.0, top_p=0.95):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            encoded_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=temperature, do_sample=do_sample, top_p=top_p)
        return self.processor.decode(output[0][inputs["input_ids"].size(-1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)