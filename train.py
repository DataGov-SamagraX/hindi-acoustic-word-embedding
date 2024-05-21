
def train():
    
    device="cuda"
    train_loader,val_loader=AudioDataset(df)
    config_file=_load_config()

    model=MultiviewModel(config)

    optim=Adam
    for epoch in range(1000):
        pass



    
