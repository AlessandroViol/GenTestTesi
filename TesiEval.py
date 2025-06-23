import json
import os
import re

from tqdm import tqdm

from Generator import Generator, QwenGenerator


def makeFiles(prompt: dict, content: str, thinking_content:str, time, model):
    os.makedirs(os.path.join("DatiTesi", model, prompt["path"]), exist_ok=True)
    with open(
        os.path.join("DatiTesi", model, prompt["path"], f"{prompt["instruction"]}_log.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        if thinking_content is not None:
            f.write(f"Generation took {time} seconds.\n\n\n\n\n" + prompt["prompt"] + "\n\n\n\n\n" + thinking_content + "\n\n\n\n\n" + content)
        else:
            f.write(f"Generation took {time} seconds.\n\n\n\n\n" + prompt["prompt"] + "\n\n\n\n\n" + content)

    code_snippets = re.findall(r"```csharp\s([^\0]*?)```", content, re.DOTALL)

    for code in code_snippets:
        code = code.strip()
        filename = str(prompt["instruction"]) + "_" + code.split("\n", 1)[0].split("/")[-1].split("\\")[-1].removesuffix(".cs").replace(":", "")

        try:
            path = os.path.join("DatiTesi", model, prompt["path"], f"{filename}.cs") 

            while os.path.exists(path):
                filename += "Dup"
                path = os.path.join("DatiTesi", model, prompt["path"], f"{filename}.cs") 

            with open(path, "w", encoding="utf-8") as f:
                f.write(code)

        except OSError as s:
            filename = str(prompt["instruction"]) + "_UnnamedFile" 
            path = os.path.join("DatiTesi", model, prompt["path"], f"{filename}.cs") 

            while os.path.exists(path):
                filename += "Dup"
                path = os.path.join("DatiTesi", model, prompt["path"], f"{filename}.cs") 

            with open(path, "w", encoding="utf-8") as f:
                f.write(code)


def processBatch(generator: Generator, prompts: dict, batch_size: int, enable_thinking: bool):
    prompt_keys = list(prompts.keys())
    for batch in tqdm([prompt_keys[i : i + batch_size] for i in range(0, len(prompt_keys), batch_size)]):
        batch_prompts=[prompts[dict_key]["prompt"] for dict_key in batch]
        answers = generator.batchGenerate(batch_prompts, enable_thinking)

        for prompt_key, answer in zip(batch, answers):
            if enable_thinking:
                thinking_content = answer["thinking_content"]
            else: 
                thinking_content = None

            content = answer["content"]
            makeFiles(prompts[prompt_key], content, thinking_content, answer["time"], model = generator._model_ref)


def processSequential(generator: Generator, prompts: dict, enable_thinking: bool):
    for dict_key in tqdm(prompts.keys()):
        prompt = prompts[dict_key]

        answer = generator.generate(prompt["prompt"], enable_thinking)

        if enable_thinking:
            thinking_content = answer["thinking_content"]
        else: 
            thinking_content = None

        content = answer["content"]

        makeFiles(prompt, content, thinking_content, answer["time"], model = generator._model_ref)

