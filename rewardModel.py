"""
description:
    训练reward模型；
    对合并后的模型进行rlhf
"""

# reward模型将通过为符合Human偏好的样本分配更高的分数来学习模仿人类的偏好

# 加载数据集
import deeplake

ds = deeplake.load('hub://genai360/Anthropic-hh-rlhf-train-set')
ds_valid = deeplake.load('hub://genai360/Anthropic-hh-rlhf-test-set')

from transformers import AutoTokenizer
# 使用不同的tokenizer，以实现对抗学习效果
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


#========================训练reward model=======================
# 对样本进行处理
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        "第一项将表示所选的（有利的）对话，而第二项将表示被拒绝的对话"
        chosen = self.dataset.chosen[idx].text()
        rejected = self.dataset.rejected[idx].text()

        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=max_length, padding='max_length')
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=max_length, padding='max_length')

        formatted_input = {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"], #对应的注意力掩码
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

      return formatted_input

# 加载数据
train_dataset = MyDataset(ds)
eval_dataset = MyDataset(ds_valid)


# iterator = iter(train_dataset)
# one_sample = next(iterator)
# print(list(one_sample.keys())) #['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected']

# 初始化reward模型和训练器
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=1
) # num_label 指定一个分数来评估序列的质量

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="DeBERTa-reward-hh_rlhf",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=20,
    weight_decay=0.001,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=1,
    bf16=True,
    logging_strategy="steps",
    logging_steps=1,
    optim="adamw_hf",
    lr_scheduler_type="linear",
    ddp_find_unused_parameters=False,
    run_name="DeBERTa-reward-hh_rlhf",
    report_to="wandb",
)

# 训练reward模型

from trl import RewardTrainer

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_length=max_length
)
print("reward traing continuing...")
trainer.train() #将自动保存检查点
print("reward training terminated...")

