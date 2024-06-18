import torch
import torch.nn as nn 
import torch.optim as optim 
from utils import save_checkpoint,load_checkpoint,_load_config
from dataset import get_loader
from model import MultiViewRNN
from loss import contrastive_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import logging
import os 

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logging.basicConfig(filename=os.path.join(logs_dir, 'training.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train(csv_file):

    config_file=_load_config()

    train_loader=get_loader(
        csv_file=csv_file,
        batch_size=config_file["batch_size"],
        loss_fn=config_file["loss_fn"])

    #flags 
    load_model=False 
    save_model=True

    writer=SummaryWriter("runs/acoustic")
    step=0

    #optimizer+model 
    model=MultiViewRNN(config_file).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config_file["lr"])

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(config_file["num_epochs"]):

        start=time.time()
        avg_loss=0
        
        for idx,batch in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):

            batch['view1_x1']=batch['view1_x1'].view(-1,batch['view1_x1'].shape[2],batch['view1_x1'].shape[1])
            
            batch['view1_x1']=batch['view1_x1'].to(DEVICE)
            batch['view2_c1']=batch['view2_c1'].to(DEVICE)
            batch['view2_c2']=batch['view2_c2'].to(DEVICE)

            output=model(batch)
            x1=output['x1']
            x2=None
            c1=output['c1']
            c2=output['c2']

            loss=contrastive_loss(obj=config_file["loss_fn"],margin=config_file["margin"],x1=x1,c1=c1,c2=c2)

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step+=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss+=loss
        
        end=time.time()
        time_taken_per_epoch=end-start
        
        avg_loss=avg_loss/len(train_loader)
        print(f"Epoch: {epoch+1}===================Avg_loss: {avg_loss:.3f}, Duration: {time_taken_per_epoch}")
        writer.add_scalar("Average loss per epoch", avg_loss, global_step=epoch)
        logging.info(f"Epoch: {epoch+1}, Avg_loss: {avg_loss:.3f}, Duration: {time_taken_per_epoch:.2f}s")

        if save_model:
            checkpoint={
                "state_dict":model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

    writer.close()
if __name__=='__main__':
    csv_file='/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/sampled_metadata.csv'
    train(csv_file)


    
