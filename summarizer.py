from transformers import pipeline

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name
        )

    def summarize(self, text, min_length=50, max_length=200):
        summary = self.summarizer(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False
        )
        return summary[0]["summary_text"]
