import torch
from torch import nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        # Freeze transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = False

        hidden_size = self.transformer.config.hidden_size

        # Linear classifier with rank-2 update
        self.W1 = nn.Linear(hidden_size, num_classes, bias=False)  # Main weight
        self.U = nn.Parameter(torch.randn(hidden_size, 2))  # Rank-2 update part 1
        self.V = nn.Parameter(torch.randn(2, num_classes))  # Rank-2 update part 2
        self.bias = nn.Parameter(torch.zeros(num_classes))  # Bias for linear layer

    def forward(self, input_ids, attention_mask):
        # Transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        # Linear layer with rank-2 update
        logits = self.W1(cls_embedding) + torch.matmul(cls_embedding @ self.U, self.V) + self.bias
        return logits
