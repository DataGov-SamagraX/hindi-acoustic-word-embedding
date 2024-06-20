import torch  
import torch.nn.functional as F 

def crossview_ap(audio_embedding,text_embedding,lev_distances):

    normalized_audio_embeddings = F.normalize(audio_embedding, p=2, dim=1)
    normalized_text_embeddings = F.normalize(text_embedding, p=2, dim=2)

    expanded_audio_embeddings = normalized_audio_embeddings.unsqueeze(1)
    
    cosine_similarities = torch.sum(expanded_audio_embeddings * normalized_text_embeddings, dim=2)

    #freeing the memory 
    del normalized_audio_embeddings
    del normalized_text_embeddings
    del expanded_audio_embeddings
    torch.cuda.empty_cache() 

    indices=torch.argsort(cosine_similarities,dim=1)

    average_precission=ranked_batch_ap(lev_distances,indices)

    del cosine_similarities
    torch.cuda.empty_cache() 

    return average_precission

def ranked_batch_ap(lev_distances, cosine_ranks):
    
    relevant_ranks = cosine_ranks.masked_select(lev_distances == 0).sort()[0]
    device=relevant_ranks.device

    pos_indices = torch.arange(1, relevant_ranks.size(0) + 1,device=device).float()

    precision_at_k = pos_indices / (relevant_ranks.float() + 1)

    batch_ap = precision_at_k.sum() / relevant_ranks.size(0)
    return batch_ap.item()