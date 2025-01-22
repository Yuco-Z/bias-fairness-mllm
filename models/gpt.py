import base64
import time
from io import BytesIO
from typing import Union

from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from PIL import Image


# build gpt class
class GPT_Model:
    def __init__(
        self,
        client: Union[OpenAI, AzureOpenAI],
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1024,
        n=1,
        patience=1000000,
        sleep_time=0,
    ):
        self.client = client
        self.model = model
        self.use_image = True if "vision" in model or "4o" in model else False
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.patience = patience
        self.sleep_time = sleep_time

    def chat_unit(self, messages, max_tokens, patience, user_prompt, caption):
        # user_prompt and caption are not used in this function
        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    n=self.n,
                )

                predictions = [choice.message.content for choice in response.choices]
                prediction = predictions[0]

                if prediction != "" and prediction is not None:
                    return prediction.strip(), user_prompt, caption

            except Exception as e:
                print(e)

                if "Please reduce the length of the messages or completion" in str(e):
                    max_tokens = int(max_tokens * 0.9)
                    print("!!Reduce max_tokens to", max_tokens)
                if max_tokens < 8:
                    return "", user_prompt, caption
                if "Please reduce the length of the messages." in str(e):
                    print("!!Reduce user_prompt to", user_prompt[:-1])
                    return "", user_prompt, caption
                if self.sleep_time > 0:
                    print(f"Sleeping for {self.sleep_time} seconds")
                    time.sleep(self.sleep_time)

        return "", user_prompt, caption

    def get_response(
        self, 
        encoded_image: Union[Image.Image, None],
        user_prompt: str,
        caption: str = None,
        seed: int = None,  # Add additional parameters here
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generate_for_answer=False, # not used
    ):
        patience = self.patience
        max_tokens = self.max_tokens

        user_messages: list[ChatCompletionContentPartParam] = []

        if self.use_image:
            if encoded_image is None:
                print(
                    f'You are using a model that supports vision: {self.model}, '
                    f'but no image was provided when generating a response. This is likely unintended.'
                )
            else:
                buffered = BytesIO()
                encoded_image.save(buffered, format="PNG")
                base64_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                user_messages.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}}
                )

        user_messages.append({"type": "text", "text": user_prompt})

        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": user_messages},
        ]

        return self.chat_unit(messages, max_tokens, patience, user_prompt, caption)

    def get_value(self, prompt_answer):
        response = []
        patience = self.patience
        max_tokens = self.max_tokens

        user_messages: list[ChatCompletionContentPartParam] = []
        user_messages.append({"type": "text", "text": prompt_answer})
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": user_messages},
        ]

        response = self.chat_unit(messages, max_tokens, patience, None, None)

        if not response:
            print(f'obtain score fail!\n')
            return []
        return [response[0]]
