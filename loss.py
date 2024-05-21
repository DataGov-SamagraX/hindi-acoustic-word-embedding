import Levenshtein
import torch 
import torch.nn.functional as F 

def edit_distance(word1,word2):
    distance=Levenshtein.distance(word1,word2)
    return distance

def embedding_loss(obj_num,margin,x1,c1,x2=None,c2=None):

    dis1=torch.mul(x1,c1)
    dis1=torch.sum(dis1,dim=1)
    
    if obj_num==0:
        if c2 is not None:
            dis2=torch.mul(x1,c2)
            dis2=torch.sum(dis2,dim=1)

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"c2 of shape {x1.shape} is required but None type object is provided")
        
        return loss 
    
    if obj_num==1:
        if c2 is not None:
            dis2=torch.mul(c1,c2)
            dis2=torch.sum(dis2,dim=1)

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"c2 of shape {c1.shape} is required but None type object is provided")
        
        return loss 
    
    if obj_num==2:
        if x2 is not None:
            dis2=torch.mul(x2,c1)
            dis2=torch.sum(dis2,dim=1)

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"x2 of shape {x1.shape} is required but None type object is provided")
        
        return loss 
    
    if obj_num==3:
        if x2 is not None:
            dis2=torch.mul(x1,x2)
            dis2=torch.sum(dis2,dim=1)

            loss= torch.mean(F.relu(margin + dis1 - dis2))
        else:
            raise ValueError(f"x2 of shape {x1.shape} is required but None type object is provided")
        
        return loss 
            

def contrastive_loss(obj,margin,x1,c1,x2=None,c2=None):
    
    for obj_num in obj:

        loss+=embedding_loss(obj_num,margin,x1,c1,x2,c2)

    return loss  

if __name__=='__main__':

    obj_num=0
    margin=0.5

    x1=torch.randn((32,1024))
    c1=torch.randn((32,1024))
    c2=torch.randn((32,1024))

    loss=embedding_loss(obj_num=obj_num,margin=margin,x1=x1,c1=c1,c2=c2)

    print(loss)
    
