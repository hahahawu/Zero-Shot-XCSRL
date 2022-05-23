import torch.nn as nn
import torch

from transformers import AutoConfig, AutoModel
import torch_scatter as scatter


class WordEncoder(nn.Module):
    def __init__(self, language_model="xlm-roberta-base", fine_tune=False, vocab_size=None, word_dropout=0.4,
                 hidden_size=512):
        super(WordEncoder, self).__init__()
        lm_config = AutoConfig.from_pretrained(language_model, output_hidden_states=True)
        self.language_model = AutoModel.from_pretrained(language_model, config=lm_config)
        if vocab_size is not None:
            self.language_model.resize_token_embeddings(vocab_size)

        if not fine_tune:
            for parameter in self.language_model.parameters():
                parameter.requires_grad = False

        word_embedding_size = 4 * lm_config.hidden_size
        self.batch_normalization = nn.BatchNorm1d(word_embedding_size)
        self.projection = nn.Linear(word_embedding_size, hidden_size)
        self.word_dropout = nn.Dropout(p=word_dropout)

        self.fine_tune = fine_tune

    def forward(self, word_ids, subword_indices=None, sequence_lengths=None, position_ids=None, attention_mask=None,
                get_pooled_output=False):
        if sequence_lengths is None and attention_mask is None:
            raise ValueError("One of sequence lengths attention mask should be assigned.")
        if attention_mask is None and sequence_lengths is not None:
            attention_mask = torch.arange(word_ids.shape[1]).unsqueeze(0).to(
                word_ids.device) < sequence_lengths.unsqueeze(1)

        if not self.fine_tune:
            with torch.no_grad():
                token_embeddings = self.language_model(input_ids=word_ids, position_ids=position_ids,
                                                       attention_mask=attention_mask)
        else:
            token_embeddings = self.language_model(input_ids=word_ids, position_ids=position_ids,
                                                   attention_mask=attention_mask)

        if get_pooled_output:
            return token_embeddings[1]

        token_embeddings = torch.cat(token_embeddings[2][-4:], dim=-1)

        token_embeddings = token_embeddings.permute(0, 2, 1)
        token_embeddings = self.batch_normalization(token_embeddings)
        token_embeddings = token_embeddings.permute(0, 2, 1)

        token_embeddings = self.projection(token_embeddings)
        word_embeddings = token_embeddings * torch.sigmoid(token_embeddings)
        token_embeddings = self.word_dropout(token_embeddings)

        if subword_indices is not None:
            word_embeddings = scatter.scatter_mean(word_embeddings, subword_indices, dim=1)
            return token_embeddings, word_embeddings
        else:
            return token_embeddings
