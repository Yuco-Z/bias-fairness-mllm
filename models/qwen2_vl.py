import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from models.base_vlm import Base_VLM_Model


config = json.load(open("./config.json"))
inf_prompts = json.load(open("./models/inf_prompts.json"))

# INF_SCALING_TEMP = "{user_prompt}\n\n{summary}\n\n{caption}\n\n{answer}"
# PRM_INF_SCALING_TEMP = "{caption}\n\n{user_prompt}"

class Qwen2vl_Model(Base_VLM_Model):
    def __init__(self, model_id, device="cuda:0", max_new_tokens=30, use_inf_scaling=False, use_prm_mcts=False, scale_at_inf=False):
        super().__init__(model_id, device, max_new_tokens, use_inf_scaling, use_prm_mcts, scale_at_inf)
        # self.max_new_tokens = max_new_tokens
        # self.device = device
        # self.model, self.processor = self.get_model(model_id)
        # self.use_inf_scaling = use_inf_scaling
        # self.use_prm_mcts = use_prm_mcts
        # self.scale_at_inf = scale_at_inf

    def get_model(self, model_id):
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, 
                                                                attn_implementation="flash_attention_2",
                                                                cache_dir=config["model_cache"]).eval().to(self.device)
        processor = AutoProcessor.from_pretrained(model_id, 
                                                  cache_dir=config["model_cache"])
        return model, processor
    
    def generate_response(self, encoded_image, user_prompt, temperature, do_sample, top_p=0.9):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        },
                        {"type": "text", "text": user_prompt},
                        ],
                        }
            ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # __import__("ipdb").set_trace()
        inputs = self.processor(text=[input_text], images=[encoded_image], 
                                padding=True, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, 
                                         temperature=temperature, do_sample=do_sample, top_p=top_p)
        generated_ids = [
        output_ids[len(input_ids): ] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return output_text

    # def get_response(self, encoded_image, user_prompt, temperature=5.0, do_sample=False, top_p=1.0):
    #     # print(f"Patience: {patience}")
    #     try:
    #         if self.use_inf_scaling:
    #             if not self.scale_at_inf:
    #                 caption = self.generate_response(encoded_image, inf_prompts["caption"], temperature, do_sample, top_p=0.5)
    #                 if not self.use_prm_mcts:
    #                     summary = self.generate_response(encoded_image, user_prompt + inf_prompts["summary"], temperature, do_sample, top_p=0.5)
    #                     steps = self.generate_response(encoded_image, user_prompt + inf_prompts["step"], temperature, do_sample, top_p=0.5)
    #                     user_prompt = INF_SCALING_TEMP.format(user_prompt=user_prompt, 
    #                                                             summary=inf_prompts["summary"] + summary, 
    #                                                             caption=inf_prompts["caption"] + caption, 
    #                                                             answer=inf_prompts["answer"] + steps)
    #                 else:
    #                     user_prompt = self.integrate_caption(user_prompt, caption) # PRM_INF_SCALING_TEMP.format(user_prompt=user_prompt, caption=caption)
    #             else:
    #                 if not self.use_prm_mcts:
    #                     user_prompt = user_prompt + inf_prompts["scale_at_infer"]
    #                 else:
    #                     user_prompt = user_prompt + inf_prompts["scale_at_infer_prm"]
    #         # __import__("ipdb").set_trace()
    #         response = self.generate_response(encoded_image, user_prompt, temperature, do_sample, top_p)
    #         # __import__("ipdb").set_trace()

    #         if verify_response(response):
    #             # print("# Verified Response")
    #             # print(response)
    #             return response, user_prompt
    #         else:
    #             print(response)
    #     except Exception as e:
    #         print(e)

    # def get_caption(self, encoded_image, temperature=1.0, do_sample=False, top_p=0.9):
    #     try:
    #         caption = self.generate_response(encoded_image, inf_prompts["caption"], temperature, do_sample, top_p)
    #         return caption
    #     except Exception as e:
    #         print(e)
    #         return None