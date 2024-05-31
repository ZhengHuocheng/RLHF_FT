# 根据本地模型，实现问答推理

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b") #确保tokenize与微调所用一致

from transformers import AutoModelForCausalLM
from accelerate import Accelerator

model = AutoModelForCausalLM.from_pretrained(
    "./OPT-RL-OrcaChat/merged", device_map={"": Accelerator().process_index}
)
model.eval()

inputs = tokenizer("Question: In one sentence, describe what the following article is about:\n\nClick on “Store” along the menu toolbar at the upper left of the screen. Click on “Sign In” from the drop-down menu and enter your Apple ID and password. After logging in, click on “Store” on the toolbar again and select “View Account” from the drop-down menu. This will open the Account Information page.  Click on the drop-down list and select the country you want to change your iTunes Store to.  You’ll now be directed to the iTunes Store welcome page. Review the Terms and Conditions Agreement and click on “Agree” if you wish to proceed. Click on “Continue” once you’re done to complete changing your iTunes Store..\n\n Answer: ", return_tensors="pt").to("cuda:0")
generation_output = model.generate(**inputs,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   max_new_tokens=128,
                                   num_beams=4,
                                   do_sample=True,
                                   top_k=10,
                                   temperature=0.6)
print( tokenizer.decode(generation_output['sequences'][0]) )