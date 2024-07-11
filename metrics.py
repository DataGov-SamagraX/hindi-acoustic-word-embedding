import torch  
import torch.nn.functional as F 

def compute_cosim(audio_emb,text_emb,var=False):
    
    sim=F.cosine_similarity(audio_emb,text_emb,dim=1)
    
    if var:
        return torch.mean(sim),torch.var(sim) 
    else:
        return torch.mean(sim)


def get_indices(audio_embedding,text_embedding):
    
    normalized_audio_embeddings = F.normalize(audio_embedding, p=2, dim=1)
    normalized_text_embeddings = F.normalize(text_embedding, p=2, dim=2)

    expanded_audio_embeddings = normalized_audio_embeddings.unsqueeze(1)

    cosine_similarities = torch.sum(expanded_audio_embeddings * normalized_text_embeddings, dim=2)

    del normalized_audio_embeddings
    del normalized_text_embeddings
    del expanded_audio_embeddings
    torch.cuda.empty_cache() 

    indices=torch.argsort(cosine_similarities, dim=1, descending=True)
    ranks = torch.zeros_like(cosine_similarities, dtype=torch.long,device=cosine_similarities.device)
    batch_size, seq_length = cosine_similarities.shape
    for i in range(batch_size):
        ranks[i,indices[i]] = torch.arange(1,seq_length+1,device=cosine_similarities.device)

    del cosine_similarities
    torch.cuda.empty_cache()

    return ranks  

def crossview_ap(audio_embedding,text_embedding,lev_distances):

    indices=get_indices(audio_embedding=audio_embedding,text_embedding=text_embedding)
    
    average_precission=ranked_batch_ap(lev_distances,indices)

    return average_precission

def ranked_batch_ap(lev_distances, cosine_ranks):

    batch_ap=0.0
    num_elements=lev_distances.size(0)

    for i in range(num_elements):
        
        relevant_ranks=cosine_ranks[i].masked_select(lev_distances[i]==0).sort()[0]
        if relevant_ranks.numel()==0:
            continue 

        pos_indices=torch.arange(1,relevant_ranks.size(0)+1,device=relevant_ranks.device).float()
        precision_at_k=pos_indices/(relevant_ranks+1)

        average_precission_i=precision_at_k.sum()/relevant_ranks.size(0)
        batch_ap+=average_precission_i
    
    if num_elements>0:
        batch_ap/=num_elements
    
    return batch_ap.item()

def crossview_corr(audio_embedding,text_embedding,lev_distances):

    indices=get_indices(audio_embedding=audio_embedding,text_embedding=text_embedding)

    spearmans_corr=ranked_batch_corr(lev_distances,indices)

    return spearmans_corr

def ranked_batch_corr(lev_distances,indices):
    
    batch_size,n=lev_distances.size()

    lev_ranks=torch.argsort(lev_distances,dim=1)
    cosine_ranks=indices 

    rank_diffs=lev_ranks-cosine_ranks
    rank_diffs_sqrt=rank_diffs**2

    num=6*torch.sum(rank_diffs_sqrt,dim=1)

    den=n*(n**2-1)

    batch_corr=1-(num/den)

    average_corr=torch.mean(batch_corr)

    return average_corr.item()

# [1,1,2,2,3,0,0,0] [9,3,4,5,2,1,8,6]----> [1,8,6] -----> ap=(p1=1/1+p2=2/8+p3=3/6)/3  precision is basically number relvant docs at that point / rank of that of that point 

if __name__=='__main__':

    pass 
