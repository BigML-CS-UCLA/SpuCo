import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast


def initialize_bert_transform(model, max_token_length):
    """
    Initializes the BERT transform for text data.

    :param model: The BERT model name.
    :type model: str
    :param max_token_length: The maximum token length for padding/truncation.
    :type max_token_length: int
    :return: The transform function.
    :rtype: Callable
    :raises ValueError: If the specified model is not recognized.
    """
    
    def get_bert_tokenizer(model):
        """
        Retrieves the appropriate BERT tokenizer based on the model name.

        :param model: The BERT model name.
        :type model: str
        :return: The BERT tokenizer.
        :rtype: PreTrainedTokenizerFast
        :raises ValueError: If the specified model is not recognized.
        """
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    tokenizer = get_bert_tokenizer(model)

    def transform(text):
        """
        Transforms the text data using BERT tokenizer.

        :param text: The input text.
        :type text: str
        :return: The transformed input as a PyTorch tensor.
        :rtype: torch.Tensor
        """
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        if model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform