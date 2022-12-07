import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from allennlp.nn.util import batched_index_select
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from typing import Optional, Tuple, Union
import copy

class BertForChID(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler", r"cls.seq_relationship.weight", r"cls.seq_relationship.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"idiom_embed_proj"]

    def __init__(self, config):
        super().__init__(config)

        # if config.is_decoder:
        #     logger.warning(
        #         "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        #         "bi-directional self-attention."
        #     )

        self.bert = BertModel(config, add_pooling_layer=False)
        
        self.cls = BertOnlyMLMHead(config)
        self.idiom_embed_proj = nn.Sequential(*[proj_resblock() for _ in range(18)])
        # import pdb
        # pdb.set_trace()
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'paris'",
    #     expected_loss=0.88,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        explainations_input_ids: Optional[torch.Tensor] = None,
        explainations_attention_mask: Optional[torch.Tensor] = None,
        explainations_token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels: torch.LongTensor of shape `(batch_size, )`
        candidates: torch.LongTensor of shape `(batch_size, num_choices, 4)`
        candidate_mask: torch.BooleanTensor of shape `(batch_size, seq_len)`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = labels.shape[0]
        explaination_length = explainations_input_ids.shape[-1]
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        explain_outputs = self.bert(
            input_ids = explainations_input_ids.reshape(batch_size*7,explaination_length),
            attention_mask=explainations_attention_mask.reshape(batch_size*7,explaination_length),
            token_type_ids=explainations_token_type_ids.reshape(batch_size*7,explaination_length),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        sequence_explain_output = explain_outputs[0][:,0,:]
        sequence_explain_output = sequence_explain_output.reshape(batch_size,7,sequence_output.shape[-1])
        # prediction_scores = self.cls(sequence_output) # (Batch_size, Seq_len, Vocab_size)  # too large
        
        masked_lm_loss = None
        
        masked_idiom = sequence_output[candidate_mask] # (Batch_size)

        candidate_idiom = self.idiom_embed_proj(self.bert.embeddings.word_embeddings(candidates))

        masked_idiom_norm = F.normalize(masked_idiom, p=2, dim=-1)
        explain_idiom_norm = F.normalize(sequence_explain_output, p=2, dim=-1) # b*num_choice*embed_dim
        candidate_idiom_norm = F.normalize(candidate_idiom, p=2, dim=-1)
        sim = torch.einsum("bf,bcf->bc", masked_idiom_norm, candidate_idiom_norm)

        sim2 = torch.einsum("bf,bcf->bc", masked_idiom_norm, explain_idiom_norm)
        candidate_final_scores = sim+sim2

        
        # masked_lm_loss = contrastive_loss(sim, labels, t=20)
        sen2idom_loss = contrastive_loss_plus(masked_idiom_norm, candidate_idiom_norm, labels, candidates, t=20)
        sen2exp_loss = contrastive_loss_plus(masked_idiom_norm, explain_idiom_norm, labels, candidates, t=20)
        # exp2idom_lm_loss = contrastive_loss_plus(explain_idiom_norm[torch.arange(batch_size),labels], candidate_idiom_norm, labels, candidates, t=20)
        masked_lm_loss = sen2exp_loss + sen2idom_loss # + exp2idom_lm_loss

        # if not return_dict:
        #     output = (prediction_scores,) + outputs[2:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=candidate_final_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def contrastive_loss(sim, labels, t=20):
    sim_exp = torch.exp(t*sim)
    positive_exp = sim_exp[torch.arange(labels.shape[0]).cuda(), labels]
    loss = (-torch.log(torch.divide(positive_exp,torch.sum(sim_exp, dim=1)))).mean()
    return loss


def contrastive_loss_plus(masked_idiom_norm, candidate_idiom_norm, labels, candidate_ids, t=20, number_choice=7):
    batch_indexs = torch.arange(labels.shape[0]).cuda()
    gt_idiom_norm = candidate_idiom_norm[batch_indexs, labels]
    sim = torch.einsum("bf,cf->bc", masked_idiom_norm, candidate_idiom_norm.reshape(-1,768))
    # word_sim = torch.einsum("bf,cf->bc", gt_idiom_norm, candidate_idiom_norm.reshape(-1,768))

    label_bias = (batch_indexs)*number_choice
    labels = labels+label_bias
    

    flatten_candidate = candidate_ids.reshape(labels.shape[0]*number_choice)
    positive_mask = ((flatten_candidate.unsqueeze(0) - flatten_candidate.unsqueeze(-1))==0).int()

    select_positive_mask = positive_mask[labels]
    select_negtive_mask = torch.ones_like(select_positive_mask).cuda() - select_positive_mask
    
    sim_exp = torch.exp(t*sim)
    # word_sim_exp = torch.exp(t*word_sim)
    positive_exp = sim_exp[batch_indexs, labels]
    # positive_word_exp = word_sim_exp[batch_indexs, labels]
    
    cross_loss = (-torch.log(torch.divide(positive_exp,  torch.sum(torch.multiply(sim_exp, select_negtive_mask), dim=1) + positive_exp  ))).mean()
    # word_loss = (-torch.log(torch.divide(positive_word_exp,  torch.sum(torch.multiply(word_sim_exp, select_negtive_mask), dim=1) + positive_word_exp ))).mean()
    loss = cross_loss # + word_loss
    return loss

class proj_resblock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.idiom_embed_proj= nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1536),
            nn.GELU(),
            nn.Linear(1536, 768)
        )

    def forward(self, x):
        x = x + self.idiom_embed_proj(x)
        return x