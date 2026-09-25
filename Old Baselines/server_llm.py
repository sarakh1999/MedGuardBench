from openai import OpenAI
from vllm import SamplingParams
import time
from concurrent.futures import ThreadPoolExecutor
import tqdm

def batchify(list_in, batch_size):
    return [list_in[i:i + batch_size] for i in range(0, len(list_in), batch_size)]

def load_url_from_log_file(log_addr):
    with open(log_addr, "r") as f:
        lines = f.readlines()
    return f"http://{lines[0].strip()}:{lines[1].strip()}/v1"

class ServerLLMOutput:
    def __init__(self, text): self.text = text
    def __getitem__(self, index): return self.text

class SeverLLMResponse:
    def __init__(self, texts): self.outputs = [ServerLLMOutput(text) for text in texts]
    def __getitem__(self, index): return self.outputs[index]

def get_response_from_server(client, messages, model_name, sampling_params, max_retries):
    retries = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                response_format={"type": "json_object"} # <--- FORCES JSON
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            retries += 1
            if retries >= max_retries: return ["ERROR"]
            time.sleep(2)

class ServerLLM:
    def __init__(self, base_url, model, max_retries=5, num_workers=1):
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.num_workers = num_workers

    def generate(self, messages_list, sampling_params=SamplingParams()):
        results = []
        for msg_batch in tqdm.tqdm(batchify(messages_list, self.num_workers)):
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(get_response_from_server, self.client, m, self.model, sampling_params, self.max_retries) for m in msg_batch]
                results.extend([f.result() for f in futures])
        return [SeverLLMResponse(r) for r in results]