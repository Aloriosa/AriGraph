import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

class BARTWrapper:
    def __init__(self, model_name: str = "facebook/bart-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts: list, max_length: int = 128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

    def generate(self, inputs: torch.Tensor, max_length: int = 128):
        return self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=1,
        )

    def fine_tune_one_example(
        self,
        input_text: str,
        target_text: str,
        lr: float = 1e-4,
        num_steps: int = 30,
    ):
        """Fine‑tune the model on a single example for a few steps."""
        self.model.train()
        tokenized = self.encode([input_text], max_length=128)
        target_tokenized = self.encode([target_text], max_length=128)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        for _ in range(num_steps):
            outputs = self.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=target_tokenized["input_ids"],
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate(self, dataset, batch_size: int = 8):
        """Return predictions and golds for the dataset."""
        self.model.eval()
        preds, golds = [], []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = self.encode(batch["input_text"])
            generated_ids = self.generate(inputs, max_length=128)
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            preds.extend(generated_texts)
            golds.extend(batch["target_text"])
        return preds, golds