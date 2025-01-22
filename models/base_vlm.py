import json
from transformers import set_seed
import random, torch
import numpy as np
import torch.backends.cudnn as cudnn

config = json.load(open("./config.json"))
inf_prompts = json.load(open("./models/inf_prompts.json"))

INF_SCALING_TEMP = "{user_prompt}\n\n{summary}\n\n{caption}\n\n{answer}"
PRM_INF_SCALING_TEMP = "{caption}\n\n{user_prompt}"

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(seed)

# build bard class
class Base_VLM_Model():
    def __init__(self, model_id, device="cuda:0", max_new_tokens=1024, use_inf_scaling=False, use_prm=False, scale_at_inf=False):
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model, self.processor = self.get_model(model_id)
        self.use_inf_scaling = use_inf_scaling
        self.use_prm = use_prm
        self.scale_at_inf = scale_at_inf

    def integrate_caption(self, user_prompt, caption):
        prompts = user_prompt.split("\n\nProblem: ")
        if len(prompts) > 1:
            return f"{prompts[0]}\n\nProblem: {caption}\n{prompts[1]}"
        else:
            return f"{caption}\n{user_prompt}"
    
    # verify response
    def verify_response(self, response):
        if isinstance(response, str):
            response = response.strip() 
        if response == "" or response == None:
            return False
        if "Response Error" in response:
            return False
        return True

    def get_model(self):
        pass
    
    def generate_response(self):
        pass

    def get_response(self, encoded_image, user_prompt, caption=None, seed=42, do_sample=True, 
                     temperature=1.0, top_p=0.95,
                     generate_for_answer=True):
        setup_seeds(seed)
        try:
            ## If use the model to generate step summary, then do not use captions
            if generate_for_answer:
                if self.use_inf_scaling:
                    if not self.scale_at_inf:
                        ## Generate caption/summarization/steps first, then integrate into user_prompt (two-step generation)
                        if caption is None:
                            caption = self.get_caption(encoded_image, do_sample=do_sample)
                        if not self.use_prm:
                            summary = self.generate_response(encoded_image, user_prompt + inf_prompts["summary"], temperature=temperature, do_sample=do_sample)
                            steps = self.generate_response(encoded_image, user_prompt + inf_prompts["step"], temperature=temperature, do_sample=do_sample)
                            user_prompt = INF_SCALING_TEMP.format(user_prompt=user_prompt,
                                                                    summary=inf_prompts["summary"] + summary, 
                                                                    caption=inf_prompts["caption"] + caption, 
                                                                    answer=inf_prompts["answer"] + steps)
                        else:
                            user_prompt = self.integrate_caption(user_prompt, caption) # PRM_INF_SCALING_TEMP.format(user_prompt=user_prompt, caption=caption)
                    else:
                        ## Give the instruction to first generate caption/summarization/steps, finally the answer (one-step generation)
                        if not self.use_prm:
                            user_prompt = user_prompt + inf_prompts["scale_at_infer"]
                        else:
                            user_prompt = user_prompt + inf_prompts["scale_at_infer_prm"]
            # __import__("ipdb").set_trace()
            response = self.generate_response(encoded_image, user_prompt, temperature, do_sample, top_p)
            torch.cuda.empty_cache()
            # __import__("ipdb").set_trace()
            if self.verify_response(response):
                return response, user_prompt, caption
            else:
                print(response)
        except Exception as e:
            print(e)

    def get_caption(self, encoded_image, temperature=1.0, do_sample=False, top_p=0.9):
        try:
            caption = self.generate_response(encoded_image, inf_prompts["caption"], 
                                             temperature=temperature, do_sample=do_sample, 
                                             top_p=top_p)
            return caption
        except Exception as e:
            print(e)
            return None