import json
import torch
from models.base_vlm import Base_VLM_Model

config = json.load(open("./config.json"))
inf_prompts = json.load(open("./models/inf_prompts.json"))

INF_SCALING_TEMP = "{user_prompt}\n\n{summary}\n\n{caption}\n\n{answer}"
PRM_INF_SCALING_TEMP = "{caption}\n\n{user_prompt}"

# build bard class
class Gemini_Model(Base_VLM_Model):
    def __init__(self, model_id, device="cuda:0", max_new_tokens=30, use_inf_scaling=False, use_prm_mcts=False, scale_at_inf=False):
        super().__init__(model_id, device, max_new_tokens, use_inf_scaling, use_prm_mcts, scale_at_inf)

    def get_model(self, model_id):
        # model_id = "lallenai/Molmo-7B-D-0924"
        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto', 
            cache_dir=config["model_cache"]
        ).eval().to(self.device)
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto', 
            cache_dir=config["model_cache"]
        )

        return model, processor
    
    def generate_response(self, encoded_image, user_prompt, do_sample=True, temperature=1.0, top_p=0.95):
        # process the image and text
        inputs = self.processor.process(
            images=[encoded_image],
            text=user_prompt,
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer, do_sample=do_sample, temperature=temperature, top_p=top_p,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response



