
from fastchat.model import get_conversation_template
from concurrent.futures import ThreadPoolExecutor
import time


from openai import AzureOpenAI, OpenAI
import anthropic

class ChatbotEngine():
    def __init__(self, generator_model) -> None:
        self.generator_model = generator_model

    def compute_usage(self):
        if "gpt-4-turbo" in self.generator_model:
            return self.total_prompt_tokens * 1e-5 + self.total_completion_tokens * 3e-5
        if "claude-3-5-sonnet" in self.generator_model:
            return self.total_prompt_tokens * 3e-6 + self.total_completion_tokens * 1.5e-5

    def query(self, conv, **kwargs):
        raise NotImplementedError

    def initialize_conversation(self):
        return get_conversation_template(self.generator_model)



class OpenAIChatbotEngine(ChatbotEngine):
    def __init__(self, generator_model) -> None:
        super().__init__(generator_model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.sleep_time = 10
        self.failed_attempts = 0
    
    def query(self, conv, temperature=0.7, repetition_penalty=1, max_new_tokens=4096, nruns=1, **kwargs):
        while True:
            try:
                prompt = conv.to_openai_api_messages()

                client = OpenAI()

                response = client.chat.completions.create(
                            model=self.generator_model,
                            messages = prompt,
                            n=nruns,
                            max_tokens=max_new_tokens,
                        )
                self.sleep_time = 10
                self.failed_attempts = 0
                break
            except Exception as e:
                print(e)
                # sleep for 10 seconds
                time.sleep(self.sleep_time)
                self.sleep_time *= 2
                self.failed_attempts += 1
                if self.failed_attempts > 10:
                    raise Exception("Too many failed attempts")
        outputs = [choice.message.content for choice in response.choices]
        if len(outputs) == 1:
            outputs = outputs[0]
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return outputs, prompt


class AzureOpenAIChatbotEngine(ChatbotEngine):
    def __init__(self, generator_model) -> None:
        super().__init__(generator_model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.sleep_time = 10
        self.failed_attempts = 0
    
    def query(self, conv, temperature=0.7, repetition_penalty=1, max_new_tokens=4096, nruns=1, **kwargs):
        while True:
            try:
                prompt = conv.to_openai_api_messages()

                client = AzureOpenAI()

                response = client.chat.completions.create(
                            model=self.generator_model,
                            messages = prompt,
                            n=nruns,
                            max_tokens=max_new_tokens,
                        )
                self.sleep_time = 10
                self.failed_attempts = 0
                break
            except Exception as e:
                print(e)
                # sleep for 10 seconds
                time.sleep(self.sleep_time)
                self.sleep_time *= 2
                self.failed_attempts += 1
                if self.failed_attempts > 10:
                    raise Exception("Too many failed attempts")
        outputs = [choice.message.content for choice in response.choices]
        if len(outputs) == 1:
            outputs = outputs[0]
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return outputs, prompt


class ClaudeChatbotEngine(ChatbotEngine):
    def __init__(self, generator_model) -> None:
        super().__init__(generator_model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.sleep_time = 10
        self.failed_attempts = 0
    
    def query(self, conv, temperature=0.7, repetition_penalty=1, max_new_tokens=4096, nruns=1, **kwargs):
        while True:
            try:
                messages = []
                for role, message in conv.messages:
                    if not message is None:
                        messages.append({"role": role, "content": message})

                client = anthropic.Anthropic()

                response = client.messages.create(
                            model=self.generator_model,
                            messages = messages,
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                self.sleep_time = 10
                self.failed_attempts = 0
                break
            except Exception as e:
                print(e)
                # sleep for 10 seconds
                time.sleep(self.sleep_time)
                self.sleep_time *= 2
                self.failed_attempts += 1
                if self.failed_attempts > 10:
                    raise Exception("Too many failed attempts")
        outputs = [choice.text for choice in response.content]
        if len(outputs) == 1:
            outputs = outputs[0]
        self.total_prompt_tokens += response.usage.input_tokens
        self.total_completion_tokens += response.usage.output_tokens
        return outputs, messages[0]["content"]
