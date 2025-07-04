import json
import os
import re

from Generator import Generator, QwenGenerator
from TesiEval import processBatch, processSequential


def runGeneration(config:dict):
    model = config["model"]
    batch_size = config["batch_size"]
    enable_thinking = config["enable_thinking"]
    name = config["name"]

    generator: Generator = QwenGenerator(model)

    if batch_size > 1:
        processBatch(generator, prompts, batch_size, enable_thinking, name)
    else:
        processSequential(generator, prompts, enable_thinking, name)


qwen8B_config = {
    "model": "Qwen/Qwen3-8B",
    "enable_thinking": True,
    "batch_size": 1,
    "name": "qwen 8"
} 

qwen14B_config = {
    "model": "Qwen/Qwen3-14B",
    "enable_thinking": True,
    "batch_size": 1,
    "name": "qwen 14"
} 

configurations = [qwen8B_config, qwen14B_config]

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

for config in configurations:
    runGeneration(config)