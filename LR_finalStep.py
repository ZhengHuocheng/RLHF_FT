'''
description: 
Version: 
Author: Zheng Huocheng
Date: 2024-05-31 15:27:16
LastEditors: Zheng Huocheng
LastEditTime: 2024-05-31 15:43:27
'''
import deeplake

ds = deeplake.load('hub://genai360/Alpaca-OrcaChat')
# ds_content: tensors=['id', 'input', 'instruction', 'output'])

# 加载微调模型时使用的tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side='left')

# 生成的数据格式应该与微调期间采用的格式保持一致
# Question: XXX\n\nAnswer:

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

      query = "Question: " + self.ds.input[idx].text() + "\n\nAnswer: "
      tokenized_question = tokenizer(query, truncation=True, max_length=400, padding='max_length', return_tensors="pt")

      formatted_input = {
        "query": query,
        "input_ids": tokenized_question["input_ids"][0],
      }

      return formatted_input

# Define the dataset object
myTrainingLoader = MyDataset(ds)


def collator(data):
    """
    整理器：将单个样本转换为batch，以传递给Trfainer
    """
    return dict((key, [d[key] for d in data]) for key in data[0])


# 用PPO模过程加载微调模型
from trl import PPOConfig

config = PPOConfig(
    task_name="OPT-RL-OrcaChat",
    steps=10_000,
    model_name="./OPT-rlhf_fine_tuned-OpenOrca/merged",
    learning_rate=1.41e-5,
    batch_size=32,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,
    ppo_epochs=4,
    seed=0,
    init_kl_coef=0.2, #确保模型不会显着偏离预训练模型
    adap_kl_ctrl=True,
    tracker_project_name="GenAI360",
    log_with="wandb",
)

from trl import set_seed
from accelerate import Accelerator

set_seed(config.seed) #设置随机状态以实现可重复性

# 存储您当前的设备 ID
current_device = Accelerator().local_process_index

# 以量化形式加载SFT模型
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

from trl import AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

#加载reward模型
from transformers import pipeline
import torch

reward_pipeline = pipeline(
    "sentiment-analysis", #定义所进行的任务
    model="./DeBERTa-v3-base-reward-hh_rlhf/checkpoint-1000",
    tokenizer="./DeBERTa-v3-base-reward-hh_rlhf/checkpoint-1000",
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    return_token_type_ids=False,
)

# 定义PPO训练器
from trl.core import LengthSampler
# 从指定范围（从定义的最小值到最大数量）中抽取样本
output_length_sampler = LengthSampler(32, 400) #(OutputMinLength, OutputMaxLength)

# 配置微调模型和reward模型的生成过程
sft_gen_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}

reward_gen_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
    "max_length": 400
}

save_freq = 50 # 检查点保存的时间间隔

from trl import PPOTrainer

ppo_trainer = PPOTrainer(
    config,
    model,
    tokenizer=tokenizer,
    dataset=myTrainingLoader,
    data_collator=collator
)

# 进入训练循环
"""
响应被解码并与提示相结合，然后再馈送到奖励模型。这使得奖励模型可以通过分配分数来评估它们与人类生成的响应的接近程度。
最后，PPO对象将根据奖励模型的得分来调整模型。
"""
from tqdm import tqdm
tqdm.pandas()

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= config.total_ppo_epochs:
        break
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **sft_gen_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = reward_pipeline(texts, **reward_gen_kwargs)

    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if save_freq and step and step % save_freq == 0:
        print("Saving checkpoint.")
        ppo_trainer.save_pretrained(f"./OPT-RL-OrcaChat/checkpoint-{step}")


# 与base model进行合并

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
  "facebook/opt-1.3b", return_dict=True, torch_dtype=torch.bfloat16
)

from peft import PeftModel

# Load the Lora model
model = PeftModel.from_pretrained(model, "./OPT-RL-OrcaChat/checkpoint-400/")
model.eval()

model = model.merge_and_unload()

model.save_pretrained("./OPT-RL-OrcaChat/merged")

