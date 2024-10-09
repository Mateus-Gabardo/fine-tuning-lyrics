from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch  # Certifique-se de importar o PyTorch para executar operações de tensor

# Diretório do modelo
model_dir = "C:/Users/gabar/Documents/colabs/modelo_finetuned/modelo_finetuned"

# Carregar o tokenizer e o modelo do diretório fornecido
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

prefix = "Classifique o seguinte canto nos tempos: natal, pascoa, quaresma, comum e advento: "

while True:
    lyric = input("Digite a letra do canto (ou 'sair' para encerrar): ")
    if lyric.lower() == 'sair':
        break

    input_text = prefix + lyric
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=32)

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Classificação: {prediction}")
