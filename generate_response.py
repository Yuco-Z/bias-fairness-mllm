import argparse
import io
import logging
import os
import sys
import time

from datasets import load_dataset
from openai import AzureOpenAI
from rich.logging import RichHandler
from tqdm import tqdm
import json
# from tasks.mmstar.build_query import create_query_data 
from rest_mcts.MCTS.task import *
from beam_search.search import VLM_Search_Engine

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        data_json = json.dumps(data, indent=4)
        f.write(data_json)


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error

def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default='mmstar')
    parser.add_argument('--test_split_name', type=str, default='val')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='./outputs/mmstar_outputs')
    parser.add_argument('--output_file', type=str, default='output_llama32_vanilla.json')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    # Local Model
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    # Remote model
    parser.add_argument(
        '--model',
        type=str,
        default='qwen25_vl_7b',
        help='llm engine',
        choices=["claude-2", "gpt4", 
                 "gpt-4-0613", "bard", 
                 "qwen25_vl_7b", "llama32_11b", 
                 "llava_ov_7b", "molmo_7b", "internvl25_8b"],
    )
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='./text_data/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='./text_data/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    parser.add_argument('--use_inf_scale', action='store_true', help='use inference scaling')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))
    # PRM MCTS
    parser.add_argument('--prm_model_id', type=str, default="peiyi9979/math-shepherd-mistral-7b-prm")
    parser.add_argument('--prompt_type', type=str, default="mistral")
    parser.add_argument('--prm_type', type=str, default="none", choices=["math_shepherd", 
                                                                         "skywork", "qwen25_vl_7b", 
                                                                         "llama32_11b", "llava_ov_7b",
                                                                         "molmo_7b", "none", "internvl25_8b"])
    parser.add_argument('--mcts_iteration_limit', type=int, default=10)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--no_rm_scoring', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--scale_at_inf', action='store_true')
    parser.add_argument('--search_type', type=str, default="none", 
                        choices=["prm_bon", "prm_beam", 
                                 "orm_bon", "orm_beam", "mcts", "none"])
    parser.add_argument('--best_of_n', type=int, default=5)
    parser.add_argument('--gen_first', action='store_true')
    parser.add_argument('--eval_gen', action='store_true', help="Evaluate in Best-of-N environment")
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--prm_step_tag', type=str, default="\n")

    args = parser.parse_args()
    return args

def set_sampling_seed():
    seed = random.randint(0, 1000000)
    return seed

def get_prm_mcts_answer(args, lm_model, prm_model, question, encoded_image, answer):
    assert args.prm_type != "none"
    task = MCTS_Task(question, encoded_image, lm_model, prm_model, prm_model.step_tag, 
                     args.prompt_type, "local", iteration_limit=args.mcts_iteration_limit, 
                     do_sample=args.do_sample, temperature=args.temperature, answer=answer,
                     top_p=args.top_p)
    output, _ = task.run()
    # __import__('ipdb').set_trace()
    return output['policy_samples'], output['value_samples'], output['user_prompt']

def get_prm_search_answer(args, search_engine, question, encoded_image):
    assert args.prm_type != "none"
    response, user_prompt = search_engine.generate_inner(encoded_image, question, 
                                                         search_type=args.search_type)
    # __import__('ipdb').set_trace()
    return response, user_prompt

def get_final_step(value_samples):
    if len(value_samples) > 2:
        final_step = value_samples[-1]["steps"][len(value_samples[-2]["steps"]):]
    else:
        final_step = value_samples[-1]["steps"]
    return final_step


def get_eval_dataset(args, config, caption_data, ocr_data):
    ######################## Evaluation Task Specific Code ########################
    if args.dataset_name.lower() == "mmstar":
        from tasks.mmstar.build_query import create_query_data
        data_list = load_dataset("Lin-Chen/MMStar", split=args.test_split_name, cache_dir=config["data_cache"])
    elif args.dataset_name.lower() == "mathvista":
        from tasks.mathvista.build_query import create_query_data
        data_list = load_dataset("AI4Math/MathVista", split=args.test_split_name, cache_dir=config["data_cache"])
    elif args.dataset_name.lower() == "mathverse":
        from tasks.mathverse.build_query import create_query_data
        data_list = load_dataset("AI4Math/MathVerse", name=args.test_split_name, 
                             split=args.test_split_name, cache_dir=config["data_cache"])
    elif args.dataset_name.lower() == "mmmu_pro":
        from tasks.mmmu.mmmu_pro.build_query import create_query_data
        data_list = load_dataset("MMMU/MMMU_Pro", name="standard (4 options)", 
                             split=args.test_split_name, cache_dir=config["data_cache"])
    elif args.dataset_name.lower() == "realworldqa":
        from tasks.realworldqa.build_query import create_query_data
        data_list = load_dataset("visheratin/realworldqa", 
                             split=args.test_split_name, cache_dir=config["data_cache"])
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    query_data, data = create_query_data(data_list, caption_data, ocr_data, args)
    ######################## Evaluation Task Specific Code ########################
    return query_data, data

def get_generation_model(args):
    if args.model == "llama32_11b":
        from models import llama32
        model = llama32.Llama32_Model("meta-llama/Llama-3.2-11B-Vision-Instruct", 
                                        args.device, args.max_new_tokens, args.use_inf_scale, 
                                        args.search_type!="none")
    elif args.model == "qwen25_vl_7b":
        from models import qwen2_vl
        model = qwen2_vl.Qwen2vl_Model("Qwen/Qwen2-VL-7B-Instruct", args.device, 
                                        args.max_new_tokens, args.use_inf_scale, 
                                        args.search_type!="none")
    elif args.model == "molmo_7b":
        from models import molmo
        model = molmo.Molmo_Model("allenai/Molmo-7B-D-0924", args.device, 
                                        args.max_new_tokens, args.use_inf_scale, 
                                        args.search_type!="none")
    elif args.model == "llava_ov_7b":
        from models import llava_onevision
        model = llava_onevision.Llava_Onevision_Model("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                    args.device, args.max_new_tokens, 
                                                    args.use_inf_scale, 
                                                    args.search_type!="none")
    elif args.model == "internvl25_8b":
        from models import internvl25
        model = internvl25.Internvl25_Model("OpenGVLab/InternVL2_5-8B", 
                                            args.device, args.max_new_tokens, 
                                            args.use_inf_scale, args.search_type!="none")
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model

def get_rm(args):
    if args.prm_type == "math_shepherd":
        from models import math_shepherd
        prm_model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        prm_model = math_shepherd.Math_shepherd_Model(prm_model_id, args.device)
    elif args.prm_type == "skywork":
        from models import skywork_prm
        prm_model_id = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        prm_model = skywork_prm.Skywork_PRM_Model(prm_model_id, step_tag=args.prm_step_tag, device=args.device)
    elif args.prm_type == "qwen25_vl_7b":
        from models import qwen2_vl
        prm_model_id = "Qwen/Qwen2-VL-7B-Instruct"
        prm_model = qwen2_vl.Qwen2vl_Model(prm_model_id, args.device, args.max_new_tokens)
    elif args.prm_type == "llama32_11b":
        from models import llama32
        prm_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        prm_model = llama32.Llama32_Model(prm_model_id, args.device, args.max_new_tokens)
    elif args.prm_type == "llava_ov_7b":
        from models import llava_onevision
        prm_model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        prm_model = llava_onevision.Llava_Onevision_Model(prm_model_id, args.device, 
                                                            args.max_new_tokens)
    elif args.prm_type == "molmo_7b":
        from models import molmo
        prm_model_id = "allenai/Molmo-7B-D-0924"
        prm_model = molmo.Molmo_Model(prm_model_id, args.device, args.max_new_tokens)

    elif args.prm_type == "internvl25_8b":
        from models import internvl25
        prm_model_id = "OpenGVLab/InternVL2_5-8B"
        prm_model = internvl25.Internvl25_Model(prm_model_id, args.device, args.max_new_tokens)
    else:
        raise ValueError(f"Unsupported PRM model: {args.prm_type}")
    return prm_model

def main(args):
    logging.info(f"{args.dataset_name}: Generating Responses - Start")
    config = json.load(open("config.json"))

    model_name = args.model #.split("/")[-1].split("-")[0]
    output_suffix = ""
    if args.do_sample:
        output_suffix = f"_sample-n{args.best_of_n}-t{args.temperature}-p{args.top_p}"
    args.output_file = f"output_{model_name}_inf_{args.use_inf_scale}_{args.prm_type}"\
        f"_{args.search_type}{output_suffix}_{args.max_num_problems}.json"

    # load data
    logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}...")

    # Convert Hugging Face data into dictionary to match local data format
    # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
    # data = {item['index']: item for item in data_list}
    logging.info("Creating new query...")
    ## Caption and OCR data are not used in MMStar
    caption_data = {}
    ocr_data = {}

    query_data, data = get_eval_dataset(args, config, caption_data, ocr_data)

    # If we were given a custom model path, load that model, otherwise use a remote service model
    if args.model:
        model = get_generation_model(args)
        
        ## Load vision-language PRM model
        if args.search_type != "none":
            if not args.do_sample:
                print("Warning: PRM searching requires do_sample=True. Setting do_sample=True.")
                args.do_sample = True

            prm_model = get_rm(args)
            if args.search_type != "mcts":
                if "prm" in args.search_type and args.no_rm_scoring:
                    logging.info(f"Using PRM search engine, setting the scoring function to True")
                    args.no_rm_scoring = False
                ## We use a different searching engine for Best-of-N search methods
                best_of_n_search_engine = VLM_Search_Engine(model, prm_model, args.device,
                                                            args.best_of_n, args.prm_type, 
                                                            step_tag=args.prm_step_tag,
                                                            do_sample=args.do_sample, 
                                                            temperature=args.temperature, 
                                                            top_p=args.top_p, 
                                                            scoring=not args.no_rm_scoring)
            else:
                best_of_n_search_engine = None

        # logging.info(f"Loading model from {args.model_path}...")
        # # TODO: Add support for local models
        # raise NotImplementedError("Local models are not yet supported.")
    else:
        model_name = args.azure_openai_model if args.azure_openai_model else args.model
        logging.info(f"Loading {model_name}...")

        if model_name == 'bard':
            from models import bard

            if args.key == '':
                logging.info("Loading key from environment variable")
                key = os.environ['_BARD_API_KEY']
            else:
                key = args.key
            model = bard.Bard_Model(key)
        elif "gpt" in model_name:
            from models import gpt

            key = args.azure_openai_api_key if args.azure_openai_api_key else args.key
            if key == '':
                key = os.getenv("OPENAI_API_KEY")

            assert (
                args.azure_openai_api_endpoint is not None
            ), "Env var AZURE_OPENAI_API_ENDPOINT is not set but is required for OpenAI client."
            assert (
                args.azure_openai_api_key is not None
            ), "Env var AZURE_OPENAI_API_KEY is not set but is required for OpenAI client."
            assert (
                args.azure_openai_api_version is not None
            ), "Env var AZURE_OPENAI_API_VERSION is not set but is required for OpenAI client."
            assert (
                args.azure_openai_model is not None
            ), "Env var AZURE_OPENAI_MODEL is not set but is required for OpenAI client."

            client = AzureOpenAI(
                azure_endpoint=args.azure_openai_api_endpoint,
                api_key=args.azure_openai_api_key,
                api_version=args.azure_openai_api_version,
            )

            model = gpt.GPT_Model(client=client, model=model_name)

        elif "claude" in model_name:
            from models import claude

            if args.key == '':
                logging.info("Loading token from environment variable")
                key = os.environ.get("ANTHROPIC_API_KEY")
            else:
                key = args.key
            model = claude.Claude_Model(model_name, key)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    logging.info(f"Model loaded.")

    full_pids = list(data.keys())

    if args.search_type == "none":
        args.output_dir = f"./outputs/{args.dataset_name}_outputs/wo_prm"
    else:
        args.output_dir = f"./outputs/{args.dataset_name}_outputs/w_prm"
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    skip_pids = []
    if not args.rerun:
        for problem_id in full_pids:
            # logging.info(f"Checking {pid}...")
            if problem_id in results and 'response' in results[problem_id]:
                response = results[problem_id]['response']
                if verify_response(response):
                    # logging.info(f"Valid response found for {pid}.")
                    skip_pids.append(problem_id)

    if len(skip_pids) > 0:
        logging.info(
            f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
        )

    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        # if args.max_num_problems <= len(test_pids):
        #     test_pids = random.sample(test_pids, args.max_num_problems)
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.warning(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    # Record the start time
    loop_start_time = time.time()

    for i, problem_id in enumerate(tqdm(test_pids)):
        problem: dict = data[problem_id].copy()

        if args.dataset_name.lower() == "mathvista":
            image_key = "decoded_image"
        elif args.dataset_name.lower() == "mmmu_pro":
            image_key = "image_1"
        else:
            image_key = "image"
        # Remove decoded Image for JSON deserialization
        problem_encoded_image = problem[image_key]
        problem.pop(image_key)

        query = query_data[problem_id]

        logging.debug("--------------------------------------------------------------")
        logging.debug(f"Generating response for problem: {problem_id}...")
        policy_samples, value_samples = None, None
        try:
            if args.search_type == "mcts":
                policy_samples, \
                    value_samples, user_prompt = get_prm_mcts_answer(args, model, 
                                                                    prm_model, query, 
                                                                    problem_encoded_image,
                                                                    data[problem_id]['answer'],)
                response = get_final_step(value_samples)
            elif args.search_type != "none":
                ## Use Best-of-N search methods
                response, user_prompt = get_prm_search_answer(args, best_of_n_search_engine, query, problem_encoded_image)
            elif args.gen_first:
                response = []
                for _ in range(args.best_of_n):
                    seed = set_sampling_seed()
                    response_cur, user_prompt, _ = model.get_response(user_prompt=query, encoded_image=problem_encoded_image, seed=seed)
                    response.append(response_cur)
            else:
                response, user_prompt, _ = model.get_response(user_prompt=query, encoded_image=problem_encoded_image)
            results[problem_id] = problem
            results[problem_id]['query'] = query
            results[problem_id]['user_prompt'] = user_prompt
            results[problem_id]['response'] = response
            if args.search_type == "mcts":
                results[problem_id]['policy_samples'] = policy_samples
                results[problem_id]['value_samples'] = value_samples

            logging.debug(f"Query: \n{query}")
            logging.debug(f"Response: \n{response}")
        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id] = problem
            results[problem_id]['error'] = str(e)
            break

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
            except Exception as e:
                logging.info(f"Error in saving {output_file_path}")
                logging.info(e)

    # Record the end time
    loop_end_time = time.time()
    # Calculate elapsed time
    elapsed_time = loop_end_time - loop_start_time
    results["decoding_time"] = "{:.6f} seconds".format(elapsed_time)
    try:
        save_json(results, output_file_path)
        logging.info(f"Saved results to {output_file_path}")
    except Exception as e:
        logging.info(f"Error in saving {output_file_path}")
        logging.info(e)

    logging.info(f"{args.dataset_name}: Generating Responses - Finish")



def eval_main(args):
    logging.info(f"{args.dataset_name}: Evaluating Responses - Start")
    config = json.load(open("config.json"))

    # load data
    logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}...")

    # Convert Hugging Face data into dictionary to match local data format
    # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
    # data = {item['index']: item for item in data_list}
    logging.info("Creating new query...")
    ## Caption and OCR data are not used in MMStar
    caption_data = {}
    ocr_data = {}

    query_data, data = get_eval_dataset(args, config, caption_data, ocr_data)
    prm_model = get_rm(args)
    logging.info(f"Model loaded.")

    if "prm" in args.search_type and args.no_rm_scoring:
        logging.info(f"Using PRM search engine, setting the scoring function to True")
        args.no_rm_scoring = False
    ## We only need our scoring model now
    best_of_n_search_engine = VLM_Search_Engine(None, prm_model, args.device,
                                                args.best_of_n, args.prm_type, 
                                                step_tag=args.prm_step_tag,
                                                do_sample=args.do_sample, 
                                                temperature=args.temperature, 
                                                top_p=args.top_p, 
                                                scoring=not args.no_rm_scoring)
 

    input_dir = f"./outputs/{args.dataset_name}_outputs/wo_prm"
    output_dir = f"./outputs/{args.dataset_name}_outputs/w_prm"
    input_file_path = os.path.join(input_dir, args.eval_file)
    output_file_path = os.path.join(output_dir, args.eval_file.split(".json")[0] + "_evaled.json")
    eval_file = read_json(input_file_path)
    test_file_len = len(eval_file)
    logging.info(f"Number of test problems to run: {len(eval_file)}")

    results = {}
    # Record the start time
    loop_start_time = time.time()

    for i, problem_id in enumerate(tqdm(eval_file.keys())):
        if problem_id == "decoding_time":
            continue
        problem_id = int(problem_id)
        problem: dict = data[problem_id].copy()

        if args.dataset_name.lower() == "mathvista":
            image_key = "decoded_image"
        elif args.dataset_name.lower() == "mmmu_pro":
            image_key = "image_1"
        else:
            image_key = "image"
        # Remove decoded Image for JSON deserialization
        problem_encoded_image = problem[image_key]
        problem.pop(image_key)

        query = query_data[problem_id]

        logging.debug("--------------------------------------------------------------")
        logging.debug(f"Generating response for problem: {problem_id}...")
        try:
            responses = eval_file[str(problem_id)]['response']
            selected_response = best_of_n_search_engine.score_judge(problem_encoded_image, problem, responses)
            ## Use Best-of-N search methods
            results[problem_id] = problem
            results[problem_id]['query'] = query
            results[problem_id]['user_prompt'] = "none"
            results[problem_id]['response'] = selected_response

            logging.debug(f"Query: \n{query}")
            logging.debug(f"Response: \n{selected_response}")
        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id] = problem
            results[problem_id]['error'] = str(e)
            break

        if (i % args.save_every == 0 and i > 0) or i == test_file_len - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
            except Exception as e:
                logging.info(f"Error in saving {output_file_path}")
                logging.info(e)

    # Record the end time
    loop_end_time = time.time()
    # Calculate elapsed time
    elapsed_time = loop_end_time - loop_start_time
    results["decoding_time"] = "{:.6f} seconds".format(elapsed_time)
    try:
        save_json(results, output_file_path)
        logging.info(f"Saved results to {output_file_path}")
    except Exception as e:
        logging.info(f"Error in saving {output_file_path}")
        logging.info(e)

    logging.info(f"{args.dataset_name}: Evaluating Responses - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    args = parse_args()
    if args.eval_gen:
        eval_main(args)
    else:
        main(args)
