import json
import os
import re

from Generator import Generator, QwenGenerator
from TesiEval import processBatch, processSequential


model = "Qwen/Qwen3-0.6B"
enable_thinking = False
batch_size = 1

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

generator: Generator = QwenGenerator(model)

if batch_size > 1:
    processBatch(generator, prompts, batch_size, enable_thinking)
else:
    processSequential(generator, prompts, enable_thinking)