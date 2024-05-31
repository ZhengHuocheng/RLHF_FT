"""
description:
    将微调后的模型与base模型进行合并
"""

# 将微调的LoRA层与base model进行合并
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
  "facebook/opt-1.3b", return_dict=True, torch_dtype=torch.bfloat16
) #base model

from peft import PeftModel

# Load the Lora model
model = PeftModel.from_pretrained(model, "./OPT-rlhf_fine_tuned-OpenOrca/<step>") #加载保存点模型

# 开启评估模式，准备合并模型
model.eval()

model = model.merge_and_unload()

model.save_pretrained("./OPT-fine_tuned-OpenOrca/merged")