import Levenshtein
import torch 
import torch.nn.functional as F 

def edit_distance(word1,word2):
    distance=Levenshtein.distance(word1,word2)
    return distance

def embedding_loss(obj_num,margin,x1,c1,x2=None,c2=None,lev_distance=None,t_max=None):

    dis1=(1.0-F.cosine_similarity(x1,c1))
    
    if obj_num==0:
        if c2 is not None and lev_distance is not None:
            dis2=(1.0-F.cosine_similarity(x1,c2))
            lev_distance=torch.tensor(lev_distance,device=x1.device)
            t_tensor=torch.full((lev_distance.size()),t_max)
            t_tensor=t_tensor.to(x1.device)
            min_tensor=(torch.min(t_tensor,lev_distance)/t_max).to(x1.device)
            margin_tensor=margin*min_tensor

            loss= torch.mean(F.relu(margin_tensor+ dis1 - dis2))
        else:
            raise ValueError(f"c2 of shape {c1.shape} is required but None type object is provided")
        
        return loss 
    
    if obj_num==1:
        if c2 is not None:
            dis2=(1.0-F.cosine_similarity(c1,c2))

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"c2 of shape {c1.shape} is required but None type object is provided")
        
        return loss 
    
    if obj_num==2:
        if x2 is not None:
            
            dis2=(1.0-F.cosine_similarity(c1,x2))

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"x2 of shape {x1.shape} is required but None type object is provided")
        
        return loss 
    
    if obj_num==3:
        if x2 is not None:
            dis2=(1.0-F.cosine_similarity(x1,x2))

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"x2 of shape {x1.shape} is required but None type object is provided")
        
        return loss 
            

def contrastive_loss(obj,margin,x1,c1,x2=None,c2=None,lev_distance=None,t_max=None):
    loss=0
    for obj_num in obj:
        loss+=embedding_loss(obj_num,margin,x1,c1,x2,c2,lev_distance,t_max)
    
    return loss  

if __name__=='__main__':

    obj_num=0
    margin=0.5
    lev_distance=[i for i in range(32)]
    t_max=9

    x1=torch.randn((32,1024))
    c1=torch.randn((32,1024))
    c2=torch.randn((32,1024))

    loss=embedding_loss(obj_num=obj_num,margin=margin,x1=x1,c1=c1,c2=c2,lev_distance=lev_distance,t_max=t_max)

    print(loss)
    
