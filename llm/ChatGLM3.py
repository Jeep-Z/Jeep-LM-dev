from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional


class ChatGLM3(LLM):
    max_token: int = 8192
    do_sample: bool = True
    temperature: float = 0.2
    tokenizer: object = None
    model: object = None
    history: List = []

    def __init__(self,):
        super().__init__()

    @property
    def _llm_type(self,):
        return "ChatGLM3"
    
    def load_model(self, model_path=None):
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, config=model_config, trust_remote_code=True, device_map="auto").eval()
        
    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        response, self.history = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        # TODO: with history

        return response
    