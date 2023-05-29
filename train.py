"""
Emotion Learning pyTorch version--Finding Visual Factors For Emotional Reaction to Abstract Painting
The MIT License (MIT)                                                                                                    
Originally created in 2023, for Python 3.x                                                                                    
Copyright (c) 2023 Diana S. Kim (diana.se.kim@gmail.com)
"""


import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Lambda
from torch import optim
import numpy as np
import pandas as pd
import presets_orig
import presets_orig_gray
from classifier import EmotionClassifier
from gram_classifier import EmotionGramClassifier
from color_gram_classifier import EmotionColorGramClassifier
from data.wiki import WikiArt

model_config={"resnet34_orig":
                     {"classifier":EmotionClassifier, "name":"resnet34","drop":1, "freeze_level":7, "mlp":[[512,100],[100,9]], "dropout":[0.3,0.3], "activation":['relu','relu']},
              "resnet34_gram":
                     {"classifier":EmotionGramClassifier, "name":"resnet34","drop":3, "freeze_level":7, "mlp":[[256+14*14,100],[100,9]], "dropout":[0.3,0.3], "activation":['relu','relu']},
"resnet34_color_gram":{"classifier":EmotionColorGramClassifier, "name":"resnet34","drop":3, "freeze_level":7, "mlp":[[256+14*14+3,100],[100,9]], "dropout":[0.3,0.3], "activation":['relu','relu']},"vgg16":
                     {"name":"vgg16","drop":2, "freeze_level":7,"mlp":[[25088,2048],[2048,1024],[1024,512],[512,9]],"dropout":[0.5,0.5,0.5,0.5],"activation":['relu','relu','relu','relu']}}


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X_color,X_gray,y,_) in enumerate(dataloader):
        
        X_color,X_gray, y = X_color.to(device), X_gray.to(device), y.to(device)
        print(X_color.size())
        print(X_gray.size())

        # Compute prediction error
        pred = model(torch.cat((X_color,X_gray),axis=1))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X_color)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X_color,X_gray,y,_) in enumerate(dataloader):
            X_color, X_gray, y = X_color.to(device), X_gray.to(device), y.to(device)
            pred = model(torch.cat((X_color,X_gray),axis=1))
            current = (batch + 1) * len(X_color)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            print(f"{current:>5d}/{size:>5d}")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

def collect(dataloader, args, model_to_collect,  device):
    try:
        os.system("mkdir "+args.do_cllct[2]) #folder to save embedding"
    except:
        print("...folder exists already")

    backbone=np.empty((0,model_config[args.model]["mlp"][0][0]))#512
    hidden_embedding=np.empty((0,model_config[args.model]["mlp"][-1][0]))#100

    for batch, (X_color,X_gray,_,_) in enumerate(dataloader):
        X_color, X_gray = X_color.to(device),X_gray.to(device)
        temp_backbone = model_to_collect.collect_feat(torch.cat((X_color,X_gray),axis=1)).detach().cpu().numpy()
        backbone=np.append(backbone,temp_backbone,axis=0)
        temp_hidden_embedding=model_to_collect.collect_hidden_embedding(torch.cat((X_color,X_gray),axis=1)).detach().cpu().numpy()
        hidden_embedding=np.append(hidden_embedding,temp_hidden_embedding,axis=0)
        print("...collect", batch)
    path=args.do_cllct[2]+"embedding_"+args.do_cllct[1]+".npz"
    np.savez(path,backbone_embedding=backbone,hidden_embedding=hidden_embedding)
    

def main(args):
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
              
    transform_train=presets_orig.ClassificationPresetTrain(crop_size=args.crop_size)
    transform_val=presets_orig.ClassificationPresetEval(crop_size=args.crop_size)
    transform_train_gray=presets_orig_gray.ClassificationPresetTrain(crop_size=args.crop_size)
    transform_val_gray=presets_orig_gray.ClassificationPresetEval(crop_size=args.crop_size)
    wikiart_train=WikiArt(args=args,img_dir=args.img_dir,transform_color=transform_train,transform_gray=transform_train_gray,split="train")
    wikiart_val=WikiArt(args=args,img_dir=args.img_dir,transform_color=transform_val,transform_gray=transform_val_gray,split="test")

    if args.do_cllct:
        train_dataloader = DataLoader(wikiart_train, batch_size=args.num_batches, shuffle=False)

    else:
        train_dataloader = DataLoader(wikiart_train, batch_size=args.num_batches, shuffle=True)#for train
    val_dataloader = DataLoader(wikiart_val, batch_size=args.num_batches, shuffle=False)


    #model
#    principle_axes=torch.from_numpy(np.load("./emotion_principle_axes/ortho_128.npz")['orthogonal_set_128']).to(device)
    model=model_config[args.model]["classifier"](name=model_config[args.model]["name"],drop=model_config[args.model]["drop"],freeze=model_config[args.model]["freeze_level"],mlp=model_config[args.model]["mlp"],dropout=model_config[args.model]["dropout"],activations=model_config[args.model]["activation"]).to(device)
    model.net.unfreeze()

              
    #optimization
    if args.data=="hist_emotion":
        model=nn.Sequential(model,nn.LogSoftmax(dim=-1))
        criterion = nn.KLDivLoss(reduction='batchmean')

        
    elif args.criterion=="max_emotion":
        criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}])
#    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)
    
    if args.do_eval:
        checkpoint=torch.load(args.do_eval,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        evaluate(val_dataloader, model, criterion, device)
        return 

    if args.do_cllct: #cllct last hidden embedding and style representation
        model.eval()
        if args.do_cllct[0] != "None": #from original model
            checkpoint=torch.load(args.do_cllct[0],map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"],strict=True)

        model_to_collect=model[0]
        if args.do_cllct[1]=="train":
           collect(train_dataloader, args, model_to_collect, device)
        elif args.do_cllct[1]=="val":#val
           collect(val_dataloader, args, model_to_collect, device)
        else:#generated
           collect(generated_dataloader, args, model_to_collect, device) 
        return

    #resume-part#
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
#        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
    
    print("start training....")
    model.train()#training
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(train_dataloader, model, criterion, optimizer, device)
#        scheduler.step()
        if epoch%5==0:
             torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},args.save_model_dir+"emotion_"+str(epoch)+".pt")
        evaluate(val_dataloader,model, criterion, device=device)
    return


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="train,eval,cllct-embedding emtion classifier")

    #data#
    parser.add_argument("--img_dir",default="/ibex/scratch/kimds/Research/P2/data/wikiart_resize/",type=str)
    parser.add_argument("--csv_dir",default="./data/",type=str) 
    parser.add_argument("--styles",default=None,nargs="*")
    parser.add_argument("--emotions",default=None,nargs="*")

              
    #model#
    parser.add_argument("--model",default="resnet34_orig", type=str)
    parser.add_argument("--crop_size",default=224,type=int)
    
    #train#
    parser.add_argument("--learning_rate",default=5e-4,type=float)#5
    parser.add_argument("--epochs",default=25,type=int)
    parser.add_argument("--num_batches",default=32,type=int)
    parser.add_argument("--start_epoch",default=0,type=int)
#    parser.add_argument("--criterion",default="kl_div",type=str)

    #option
    parser.add_argument("--save_model_dir",default="./model/",type=str)#train
    parser.add_argument("--do_cllct",default=None,nargs="*")#collect-embedding ["model.pt","val","./cllct_embedding/"]
    parser.add_argument("--resume",default=None,type=str,help="model dir to resume training")#resume-training
    parser.add_argument("--do_eval",default=None,type=str,help="model dir to eval")#just-evaluation
    

    #test
    parser.add_argument("--version",default=None,type=str)
    parser.add_argument("--color",default=None,type=str,help="choose color or gray")
    parser.add_argument("--data",default="hist_emotion",type=str,help="kl_div or softmax cross entropy")

    return parser

if __name__== "__main__":
    args = get_args_parser().parse_args()
    main(args)
