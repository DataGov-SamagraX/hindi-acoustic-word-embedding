import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR 
from utils import save_checkpoint,load_checkpoint,_load_config
from dataset import get_train_loader,get_dev_loader
from model import MultiViewRNN
from loss import contrastive_loss
from metrics import crossview_ap,crossview_corr,compute_cosim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import logging
import os 
import pytorch_warmup as warmup 

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logging.basicConfig(filename=os.path.join(logs_dir, 'training.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train(train_csv_file,dev_csv_file):

    config_file=_load_config()

    train_loader=get_train_loader(
        csv_file=train_csv_file,
        batch_size=config_file["train_batch_size"],
        loss_fn=config_file["loss_fn"],
        max_mfcc_len=config_file['max_mfcc_train'])

    dev_loader=get_dev_loader(
        csv_file=dev_csv_file,
        batch_size=config_file["dev_batch_size"],
        max_mfcc_len=config_file['max_mfcc_dev']
    )
    #flags 
    load_model=False
    save_model=True

    writer=SummaryWriter("runs/acoustic")
    step=0
    dev_step=0

    #optimizer+model 
    model=MultiViewRNN(config_file).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config_file["lr"], betas=(config_file['beta1'],config_file['beta2']), weight_decay=config_file['weight_decay'])
    
    #lr scheduler 
    steps_per_epoch=len(train_loader)
    warmup_period=config_file['warmup_period']
    num_steps=steps_per_epoch*config_file['num_epochs']-warmup_period
    t0=num_steps//3
    lr_min=3e-5
    max_steps=t0*3+warmup_period

    lr_scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=1, eta_min=lr_min
    )

    warmup_sheduler=warmup.LinearWarmup(optimizer,warmup_period)


    if load_model:
        checkpoint_path='/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/checkpoints4/checkpoint_epoch_34.pth.tar'
        step = load_checkpoint(torch.load(checkpoint_path), model, optimizer)


    for epoch in range(config_file["num_epochs"]):
        
        model.train()
        start=time.time()
        avg_loss=0
        cosim_train_neg=0
        for idx,batch in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):

            batch['view1_x1']=batch['view1_x1'].view(-1,batch['view1_x1'].shape[2],batch['view1_x1'].shape[1])
            batch['view1_x2']=batch['view1_x2'].view(-1,batch['view1_x2'].shape[2],batch['view1_x2'].shape[1])
            
            batch['view1_x1']=batch['view1_x1'].to(DEVICE)
            batch['view1_x2']=batch['view1_x2'].to(DEVICE)
            batch['view2_c1']=batch['view2_c1'].to(DEVICE)
            batch['view2_c2']=batch['view2_c2'].to(DEVICE)
            lev_distance=batch['edit_distance']

            output=model(batch)
            x1=output['x1']
            x2=output['x2']
            c1=output['c1']
            c2=output['c2']

            loss=contrastive_loss(obj=config_file["loss_fn"],
                                  margin=config_file["margin"],
                                  x1=x1,
                                  c1=c1,
                                  c2=c2,
                                  x2=x2,
                                  lev_distance=lev_distance,
                                  t_max=config_file['t_max'])

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step+=1
            
            #Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #lr schedulers
            with warmup_sheduler.dampening():
                if warmup_sheduler.last_step+1>=warmup_period:
                    lr_scheduler.step()
            if warmup_sheduler.last_step+1>=max_steps:
                break 
            lr=optimizer.param_groups[0]['lr']
            writer.add_scalar("lr vs steps",lr,global_step=step)

            #other metrics
            cos_sim_pos=compute_cosim(x1,c1)
            cos_sim_neg=compute_cosim(x1,c2)
            cosim_train_neg+=cos_sim_neg
            writer.add_scalar("Cosine Similarity gt Vs Step",cos_sim_pos.item(), global_step=step)
            writer.add_scalar("Cosine similarity neg Vs Step",cos_sim_neg.item(), global_step=step)
            
            avg_loss+=loss

        
        avg_loss=avg_loss/len(train_loader)
        average_cosim_neg=cosim_train_neg/len(train_loader)
        writer.add_scalar("Average train negative similarity Vs Epoch", average_cosim_neg, global_step=epoch)
        writer.add_scalar("Average train loss Vs Epoch", avg_loss, global_step=epoch)
        end=time.time()
        time_taken_per_epoch=end-start


        print("Evaluating Dev")
        model.eval()
        average_precision=0.0
        average_corr=0.0
        cosim_pos_dev=0.0
        cosim_neg_dev=0.0
        avg_dev_loss=0.0
        for idx,batch in tqdm(
                enumerate(dev_loader), total=len(dev_loader), leave=False
            ):
            
                mfcc=batch["mfcc"]
                mfcc=mfcc.view(-1,mfcc.shape[2],mfcc.shape[1])
                mfcc=mfcc.to(DEVICE)
                mfcc_input={"view1_x1":mfcc}
                audio_emb=model(mfcc_input)["x1"] 

                input_text_tensor=batch["sampled_one_hot"]
                batch_size=input_text_tensor.shape[0]
                sampled_shape=input_text_tensor.shape[1]
                input_text_tensor=input_text_tensor.view(input_text_tensor.shape[0]*sampled_shape,
                                                        input_text_tensor.shape[2],
                                                        input_text_tensor.shape[3])
                input_text_tensor=input_text_tensor.to(DEVICE)
                input_one_hot={"view2_c1":input_text_tensor}
                out_one_hot=model(input_one_hot)["c1"]
                text_emb=out_one_hot.view(batch_size,
                                        sampled_shape,
                                        out_one_hot.shape[1])
            
                lev_distances=batch["lev_scores"].to(DEVICE)
            
                ranked_ap=crossview_ap(audio_embedding=audio_emb,
                                    text_embedding=text_emb,
                                    lev_distances=lev_distances)
                ranked_corr=crossview_corr(audio_embedding=audio_emb,
                                        text_embedding=text_emb,
                                        lev_distances=lev_distances)
                average_precision+=ranked_ap
                average_corr+=ranked_corr

                #calculating cosine sim for dev with ground truth 
                ground_truth=batch['ground_truth']
                ground_truth=ground_truth.to(DEVICE)
                gt_input={'view2_c1':ground_truth}
                gt_emb=model(gt_input)['c1']

                negative_text=batch['one_hot_c2']
                negative_text_sample=negative_text.to(DEVICE)
                negative_text_input={'view2_c2':negative_text_sample}
                neg_text_emb=model(negative_text_input)['c2']


                cossim_pos_dev_batch,cossim_pos_var_batch=compute_cosim(audio_emb=audio_emb,text_emb=gt_emb,var=True)
                cossim_neg_dev_batch=compute_cosim(audio_emb=audio_emb,text_emb=neg_text_emb)
                cosim_pos_dev+=cossim_pos_dev_batch
                cosim_neg_dev+=cossim_neg_dev_batch

                #calculating dev loss 
                neg_audio=batch['mfcc_x2']
                neg_audio=neg_audio.view(-1,batch['mfcc_x2'].shape[2],batch['mfcc_x2'].shape[1])
                neg_audio=neg_audio.to(DEVICE)
                neg_audio_input={'view1_x2':neg_audio}
                neg_audio_emb=model(neg_audio_input)["x2"]

                loss_lev_distance=batch['edit_distance']

                dev_loss=contrastive_loss(obj=config_file["loss_fn"],
                                  margin=config_file["margin"],
                                  x1=audio_emb,
                                  c1=gt_emb,
                                  c2=neg_text_emb,
                                  x2=neg_audio_emb,
                                  lev_distance=loss_lev_distance,
                                  t_max=config_file['t_max'])
                avg_dev_loss+=dev_loss

                writer.add_scalar("Eval loss", dev_loss.item(), global_step=dev_step)
                dev_step+=1
                
                writer.add_scalar("Dev Variance(cosine similarity) Vs Step",cossim_pos_var_batch.item(), global_step=dev_step)
                torch.cuda.empty_cache()


            
        cosim_pos_dev=cosim_pos_dev/len(dev_loader)
        cosim_neg_dev=cosim_neg_dev/len(dev_loader)

        average_precision=average_precision/len(dev_loader)
        average_corr=average_corr/len(dev_loader)

        avg_dev_loss+=avg_dev_loss/len(dev_loader)

        #scheduler.step(average_precision)
        
        writer.add_scalar("Cosine Similarity pos dev Vs Epoch",cosim_pos_dev, global_step=epoch)
        writer.add_scalar("Cosine Similarity neg dev Vs Epoch",cosim_neg_dev, global_step=epoch)

        writer.add_scalar("Average precission Vs Epoch", average_precision, global_step=epoch)
        writer.add_scalar("Average Spearman's corr Vs Epoch", average_corr, global_step=epoch)

        writer.add_scalar("Average eval loss Vs Epoch", avg_dev_loss, global_step=epoch)

        if average_precision and average_corr:
            logging.info(f"Epoch: {epoch+1}, Avg_loss: {avg_loss:.3f}, Duration: {time_taken_per_epoch:.2f}s, Average_precission: {average_precision:.3f}, Average_corr: {average_corr:.3f}")
            print(f"Epoch: {epoch+1}===================Avg_loss: {avg_loss:.3f}, Duration: {time_taken_per_epoch}, Average_P: {average_precision:.3f}, Average_corr: {average_corr:.3f}")

        """else:
            logging.info(f"Epoch: {epoch+1}, Avg_loss: {avg_loss:.3f}, Duration: {time_taken_per_epoch:.2f}s")
            print(f"Epoch: {epoch+1}===================Avg_loss: {avg_loss:.3f}, Duration: {time_taken_per_epoch}")"""

        if save_model:
            checkpoint={
                "state_dict":model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        """if scheduler.optimizer.param_groups[0]['lr'] < 1e-18:
            print("Learning rate reached the minimum threshold. Stopping training.")
            break"""
        
        writer.close()


if __name__=='__main__':
    train_csv_file='/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_train.csv'
    dev_csv_file='/home/ubuntu/acoustic_stuff/hindi-acoustic-word-embedding/dataset/train_aligned_dataset/sample_bhashini_dev.csv'
    train(train_csv_file,dev_csv_file)







