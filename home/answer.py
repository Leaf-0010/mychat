from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model_path = 'model'
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)


# inference
def generator(question):
    input_ids = tokenizer(
        "python是谁创造的？", return_tensors="pt"
    ).input_ids  # Batch size 1
    outputs = model.generate(input_ids,max_length=256)
    print(outputs.shape)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    return answer
# studies have shown that owning a dog is good for you.