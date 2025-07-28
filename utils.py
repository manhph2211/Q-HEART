import torch
import numpy as np
from nltk.tokenize import TreebankWordTokenizer


def treebank_tokenize(s):
    return TreebankWordTokenizer().tokenize(s)


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    generated=None,
    entry_length=65,
    temperature=1.0,
):
    model.eval()
    stop_token_index = tokenizer.eos_token_id
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        for i in range(entry_length):
            outputs = model.llm(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            logits = logits.softmax(-1).log()
            # final_logit

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            if model.llm_type in ["google/gemma-2-2b", "google/gemma-2b", "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct"]: 
                if model.setting == "lora":
                    next_token_embed = model.llm.base_model.model.model.embed_tokens(
                        next_tokens.squeeze()
                    ).view(generated.shape[0], 1, -1)
                else:
                    next_token_embed = model.llm.model.embed_tokens(
                        next_tokens.squeeze()
                    ).view(generated.shape[0], 1, -1)
            elif model.llm_type == "gpt2":
                next_token_embed = model.llm.transformer.wte(
                    next_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.llm.get_input_embeddings()(tokens[:,-1])
                next_token_embed=next_token_embed.squeeze().view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts