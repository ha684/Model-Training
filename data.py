def analyze_token_lengths(dataset, tokenizer, instruction, num_samples=None):
    """
    Phân tích độ dài token của dataset
    """
    token_lengths = []
    
    samples_to_check = range(len(dataset)) if num_samples is None else range(min(num_samples, len(dataset)))
    
    for idx in samples_to_check:
        sample = dataset[idx]
        
        # Tạo messages giống như khi training
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["text"]}
            ]}
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        
        # Tokenize với cả image và text
        inputs = tokenizer(
            images=sample["image"],
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        
        # Lấy số lượng token
        num_tokens = inputs['input_ids'].shape[1]
        token_lengths.append(num_tokens)
        
        if idx < 3:  # Print chi tiết cho 3 sample đầu
            print(f"\n--- Sample {idx} ---")
            print(f"Total tokens: {num_tokens}")
            print(f"Text output length: {len(sample['text'])} characters")
    
    return token_lengths

# Chạy phân tích
token_lengths = analyze_token_lengths(train_dataset, tokenizer, instruction, num_samples=50)

# Thống kê
import numpy as np
print("\n=== TOKEN LENGTH STATISTICS ===")
print(f"Min: {np.min(token_lengths)}")
print(f"Max: {np.max(token_lengths)}")
print(f"Mean: {np.mean(token_lengths):.2f}")
print(f"Median: {np.median(token_lengths):.2f}")
print(f"95th percentile: {np.percentile(token_lengths, 95):.2f}")
print(f"99th percentile: {np.percentile(token_lengths, 99):.2f}")