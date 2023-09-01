from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
#from modeling_chatglm import ChatGLMForConditionalGeneration
#from tokenization_chatglm import ChatGLMTokenizer
import torch
from loguru import logger

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda:0"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        with torch.cuda.device("cuda:1"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 


tokenizer =AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b", #cache_dir = '.',
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
        "THUDM/chatglm2-6b", #cache_dir = '.',
        device_map="auto",
        trust_remote_code=True).half()#.to('cuda:0')
#model.to('cuda:0')
#model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", cache_dir = '/mntnlp/qian.lwq/Chatglm_t',trust_remote_code=True).half()

class ChatGLM(LLM):
    max_token: int = 8192
    temperature: float = 0.1
    top_p = 0.9
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM2-6b"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        #model= ChatGLMForConditionalGeneration.from_pretrained("/mntnlp/qian.lwq/Chatglm_t/models--THUDM--chatglm-6b/snapshots/chatglm",trust_remote_code=True).half().cuda()
        logger.debug(f"prompt={prompt}")
        response, updated_history = model.chat(
            tokenizer,
            prompt,
            history=[] ,   #self.history, 由于langchain的memroy已经会将human:ai格式的历史写到prompt  所以这里最好是history=[]
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        if len(updated_history[-1]) > 0 :
            logger.error(f"updated_history[-1]完整部分包括memory_history,context和question={updated_history[-1]}")
            # template = """现提供如下信息:
            #            history={memory_history}
            #            context={context}
            #            请使用上述信息回答:{question}"""  
            
            #self.history = [(updated_history[-1][0][-50:] ,updated_history[-1][1])]  # 因为history结构为[(q,a),(q,a)...] 所以这种写法是强行将history最后一个元组的最后50字作为q 不优雅 
            self.history = [(updated_history[-1][0].split("请使用上述信息回答:")[-1] ,updated_history[-1][1])] #这种写法是根据template格式来切history最后的问题部分 我们试试
        else :
            self.history = []
        logger.error(f"history[-1]仅仅包含最后的question部分,history={self.history}\nlen(self.history)={len(self.history)}")
        logger.error(f"char_len_total ={sum([len(item[0])+len(item[1]) for item in self.history])}") #因为history结构为[(q,a),(q,a)...]
        for item in self.history:
            logger.debug(f"item in history={item}")
        return response
