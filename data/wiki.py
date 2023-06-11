import os
import unicodedata
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image #compare
from PIL import Image
import emotions
import styles


#dist
def img_distribution(g):
    image_distribution = np.zeros(9, dtype=np.float32)
    for l in g.emotion_label:
        image_distribution[l] += 1.0
    return image_distribution/sum(image_distribution)

##max emotion
def img_max_emotion(g):
    image_distribution = np.zeros(9, dtype=np.float32)
    for l in g.emotion_label:
        image_distribution[l] += 1.0
    return np.argmax(image_distribution)


class WikiArt(Dataset):
    def __init__(self,args,img_dir,transform_color=None, transform_gray=None,split='train'):
        self.df = pd.read_csv("../../artemis_official/"+args.version+"/artemis_preprocessed.csv",header=0)
        #self.df=self.df[df.art_style.isin(args.emotions)]
        self.df=self.df[self.df.emotion.isin(emotions.ARTEMIS_EMOTIONS_9)]
        self.df=self.df[self.df.art_style.isin(styles.ABSTRACT_STYLES)]

        def update_emotion_label(x):
#            return args.emotions.index(x)
            return emotions.ARTEMIS_EMOTIONS_9.index(x)

        self.df["emotion_label"]=self.df["emotion"].apply(update_emotion_label)
        self.df =self.df[(self.df.split==split)]#train or test   
        
        #subsets#
        if args.data=="hist_emotion":
            self.df=self.df.groupby(['art_style','painting','split']).apply(img_distribution).to_frame('emotion_label').reset_index()
        elif args.data=="max_emotion":
            self.df=self.df.groupby(['art_style','painting','split']).apply(img_max_emotion).to_frame('emotion_label').reset_index()
        else:#as is
            self.df=self.df[['art_style','painting','split','emotion_label']]
        
        #reset index
        self.df.reset_index(inplace=True, drop=True)
        self.img_dir = img_dir
        self.transform_color = transform_color
        self.transform_gray = transform_gray

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx,0],self.df.iloc[idx,1]+".jpg")
        img_path = unicodedata.normalize('NFD', img_path)
        image = Image.open(img_path)
        label = self.df.iloc[idx,3]
        try:
            image_color = self.transform_color(image)
            image_gray = self.transform_gray(image)
        except:
            print(img_path)
        return image_color, image_gray, label, img_path
