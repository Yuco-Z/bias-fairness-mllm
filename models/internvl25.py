import json
import torch
from transformers import AutoModel, AutoTokenizer
from models.base_vlm import Base_VLM_Model
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

config = json.load(open("./config.json"))
inf_prompts = json.load(open("./models/inf_prompts.json"))

INF_SCALING_TEMP = "{user_prompt}\n\n{summary}\n\n{caption}\n\n{answer}"
PRM_INF_SCALING_TEMP = "{caption}\n\n{user_prompt}"

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# build bard class
class Internvl25_Model(Base_VLM_Model):
    def __init__(self, model_id, device="cuda:0", max_new_tokens=30, use_inf_scaling=False, use_prm_mcts=False, scale_at_inf=False):
        super().__init__(model_id, device, max_new_tokens, use_inf_scaling, use_prm_mcts, scale_at_inf)

    def get_model(self, model_id):
        # model_id = "OpenGVLab/InternVL2_5-8B"
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            cache_dir=config["model_cache"]
            ).eval().to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, 
                                                  use_fast=False, cache_dir=config["model_cache"])
        return model, tokenizer
    
    def generate_response(self, encoded_image, user_prompt, do_sample=True, temperature=1.0, top_p=0.95):
        pixel_values = load_image(encoded_image, max_num=12).to(torch.bfloat16).to(self.device)
        generation_config = dict(max_new_tokens=1024, do_sample=do_sample, temperature=temperature, top_p=top_p)

        question = f'<image>\n{user_prompt}'
        response = self.model.chat(self.processor, pixel_values, question, generation_config)
        return response
