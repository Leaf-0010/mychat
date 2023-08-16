
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
model_path = 'model'

tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)

#model = MT5ForMultimodalGeneration.from_pretrained(model_path, padding_idx=padding_idx)
model.resize_token_embeddings(len(tokenizer))

input_sequences = ['Elasticsearch 支持哪些数据类型？']


encoding = tokenizer(input_sequences,
                     padding="max_length",
                     max_length=256,
                     truncation=True,
                     return_tensors="pt")

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask



print(tokenizer.decode(model.generate(input_ids)[0], skip_special_tokens=True))