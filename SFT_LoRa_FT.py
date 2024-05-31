"""
description:
    利用带Lora的SFT微调OPT模型
"""
import deeplake
from data_precess import prepare_sample_text
# 加载deeplake中的OpenOrca数据集

ds = deeplake.load('hub://genai360/OpenOrca-1M-train-set')
ds_valid = deeplake.load('hub://genai360/OpenOrca-1M-valid-set')
# ds_content: tensors=['id', 'question', 'response', 'system_prompt'])
# question由LLM生成的查询，response是由模型生成的相应，system_prompt是用来指导模型生成上下文的指令

# 加载OPT模型分词器
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# 定义样本格式限制
from trl.trainer import ConstantLengthDataset

train_ds = ConstantLengthDataset(
    tokenizer,
    ds,
    formatting_func=prepare_sample_text,
    infinite=True,
    seq_length=2048    
)

eval_ds = ConstantLengthDataset(
    tokenizer,
    ds_valid,
    formatting_func=prepare_sample_text,
    seq_length=1024
)

train_dataset.start_iteration = 0
iterator = iter(train_dataset)
sample = next(iterator) #{'input_ids': tensor([ 16, 358, 828,  ..., 137,  79, 362]), 'labels': tensor([ 16, 358, 828,  ..., 137,  79, 362])}

# 利用LoRA微调LLM，以保持低的内存需求
from peft import LoraConfig
# 配置LoRA微调方式
lora_config = LoraConfig(
    r=8, # 低秩近似参数，降低矩阵大小，控制模型的复杂度,
    alpha=32, # 控制正则化强度的参数，控制模型对训练数据的拟合程度。
    lora_dropout=0.05, # 控制LoRA层中dropout的参数，控制模型的泛化能力。
    bias="none", # 控制是否使用偏置的参数，控制模型的性能。
    take_type="CAUSAL_LM",  # 推理模型类型
)

# 配置训练超参数
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./OPT-rlhf_fine_tuned-OpenOrca",
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=2,
    eval_steps=2000,
    save_steps=2000,
    logging_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    gradient_accumulation_steps=1,
    bf16=True, # 训练过程参数使用bfloat16格式进行
    weight_decay=0.05,
    ddp_find_unused_parameters=False,
    run_name="OPT-rlhf_fine_tuned-OpenOrca",
    report_to="wandb",
)
# 配置量化超参数，执行量化过程
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True, # 以 4 bit格式加载模型
   bnb_4bit_quant_type="nf4", #采用nb4格式实现权重的嵌套量化
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16 # 训练过程参数使用bfloat16格式进行
)



# 加载OPT模型的预训练权重
from transformers import AutoModelForCausalLM
from accelerate import Accelerator

model = AutoModelForCausalLM.from_pretrained(
  "facebook/opt-1.3b",
	quantization_config=quantization_config,
	device_map={"": Accelerator().process_index}
)

# 调整网络架构，提高效率
from torch import nn

for param in model.parameters():
  param.requires_grad = False #冻结
  if param.ndim == 1:
    param.data = param.data.to(torch.float32) #转换为32位全精度

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# 使用带Lora的SFT微调模型，使用RLHF训练数据进行训练
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,  #训练配置
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_config, #lora配置
    packing=True,
)

# 开始微调训练
print("SFT with Lora training continuing...")
trainer.train()
print("SFT with Lora training terminated")
