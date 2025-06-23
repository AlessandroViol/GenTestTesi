from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from time import time
import os

class Generator(ABC):
    def __init__(self, model):
        pass
    
    @abstractmethod
    def generate(self, prompt: str, enable_thinking: bool):
        pass
    
    @abstractmethod   
    def batchGenerate(self, prompts: list[str], batch_size: int, enable_thinking: bool):
        pass
    

class QwenGenerator(Generator):
    def __init__(self, model):
        self._model_ref = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_ref, 
            local_files_only=False,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_ref, 
            trust_remote_code=True, 
            local_files_only=False, 
            torch_dtype=torch.float16, 
            device_map="auto",
        ).eval()


    def generate(self, prompt: str, enable_thinking: bool):
        print(f"Starting generation")
        start = time()
        messages = [{"role": "user", "content": prompt}]

        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        with torch.inference_mode():
            model_input = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(**model_input, max_new_tokens=32768)
            output_ids = generated_ids[0][len(model_input.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        end = time()
        print(f"Response found in {end-start} seconds")

        if enable_thinking:
            return {"thinking_content": thinking_content , "content": content, "time": end-start}
        else:
            return {"content": content, "time": end-start}
        
    
    def batchGenerate(self, prompts: list[str], enable_thinking: bool):
        if not prompts:
            return []

        print(f"Starting generation for {len(prompts)} prompts...")
        start_time = time()

        batched_input_texts = []
        for prompt_content in prompts:
            messages = [{"role": "user", "content": prompt_content}]
           
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            batched_input_texts.append(text)

        with torch.inference_mode():
            model_inputs = self.tokenizer(
                batched_input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
                padding_side="left"
            ).to(self.model.device)

            generated_ids_batch = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id 
            )

        end_time = time()
        total_batch_time = end_time - start_time
        print(f"{len(prompts)} responses generated in {total_batch_time:.2f} seconds.")

        results = []
        qwen_thinking_separator_token_id = 151668

        for i in range(len(prompts)):
            input_actual_length = model_inputs.attention_mask[i].sum().item()

            output_ids_tensor = generated_ids_batch[i][input_actual_length:]
            output_ids = output_ids_tensor.tolist()

            thinking_content = ""
            content = ""

            try:
                idx_of_separator_from_end = output_ids[::-1].index(qwen_thinking_separator_token_id)
                split_point = len(output_ids) - idx_of_separator_from_end
                
                thinking_content = self.tokenizer.decode(output_ids[:split_point], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[split_point:], skip_special_tokens=True).strip("\n")

            except ValueError: 
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                if enable_thinking: 
                    thinking_content = ""


            item_result = {"time": total_batch_time} 
            if enable_thinking:
                item_result["thinking_content"] = thinking_content
                item_result["content"] = content
            else:
                item_result["content"] = content
            results.append(item_result)

        return results
    