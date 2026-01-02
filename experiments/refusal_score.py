import torch

def token_ids_first_piece(tokenizer, strings):
    ids = []
    for s in strings:
        toks = tokenizer(s, add_special_tokens=False).input_ids
        if len(toks) == 0:
            continue
        ids.append(toks[0])
    # unique + stable
    return sorted(set(ids))

def refusal_score_from_logits(logits_last_pos, ref_ids, ok_ids):
    # logits_last_pos: [vocab]
    ref = torch.logsumexp(logits_last_pos[ref_ids], dim=0)
    ok  = torch.logsumexp(logits_last_pos[ok_ids], dim=0)
    return (ref - ok).item()
