def prepare_sample_text(example):
    """
    从dataset中处理文本数据
    """
    text = f"Question: {example['question'][0]}\n\nAnswer: {example['response'][0]}"
    return text