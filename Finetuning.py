from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Laden eines Datensatzes
dataset = load_dataset("path_to_your_data.json")

# Tokenizer laden
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenisierung der Eingabedaten
def tokenize_function(examples):
    return tokenizer(examples["input"], examples["output"], truncation=True, padding="max_length")

tokenized_datasets = dataset.map(tokenize_function, batched=True)
#Modell laden und Feintuning durchführen:
# Modell laden
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Trainingseinrichtungen definieren
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Feintuning starten
trainer.train()

#Modell speichern:
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
