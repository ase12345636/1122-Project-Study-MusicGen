import os
import pandas as pd 
import random
from pydub import AudioSegment
import json


def check_data():
    
    df = pd.read_csv("dataset.csv")
    ytid = df['ytid'].values.tolist()
    col = df.columns.values
    yt = []
    start = []
    end = []
    apl = []
    al = []
    cap = []
    id = []
    ib = []
    ia = []
    path = "music_data"
    items = os.listdir(path)
    for i in items:
        base=os.path.basename(i)
        file=os.path.splitext(base)[0]
        # print(file)
        if file in ytid:
            idx = ytid.index(file)
            yt.append(df.at[idx,col[0]])
            start.append(df.at[idx,col[1]])
            end.append(df.at[idx,col[2]])
            apl.append(df.at[idx,col[3]])
            al.append(df.at[idx,col[4]])
            cap.append(df.at[idx,col[5]])
            id.append(df.at[idx,col[6]])
            ib.append(df.at[idx,col[7]])
            ia.append(df.at[idx,col[8]])
    dict = {col[0]:yt, col[1]:start, col[2]:end,
            col[3]:apl, col[4]:al, col[5]: cap,
            col[6]:id, col[7]:ib, col[8]: ia}
    print(len(yt))
    newfile = pd.DataFrame(dict)
    newfile.to_csv('accept_data.csv')


def rename_file():
    
    df = pd.read_csv("accept_data.csv")
    col = df.columns.values
    ytid = df['ytid'].values.tolist()
    name=[]
    key=[]
    cap=[]
    path = 'music/'
    items = os.listdir(path)
    count = 1
    for i in items:
        base=os.path.basename(i)
        file=os.path.splitext(base)[0]
        if file in ytid:
            idx = ytid.index(file)
            new_name="Music_"+str(count)+".wav"
            os.rename(path+i,path+new_name)
            name.append(new_name)
            key.append(df.at[idx,col[5]])
            cap.append(df.at[idx,col[6]])
            print("rename: "+(i)+"  --->  "+ new_name + " ok!")
        count+=1
        
    dict = {"name":name,"keywords":key,"caption":cap}
    newfile = pd.DataFrame(dict)
    newfile.to_csv('rename_data.csv',index=False)
    

def cut_music():
    
    dir = 'music/'
    items = os.listdir(dir)
    for file in items:
        fullpath = dir + file
        sound = AudioSegment.from_wav(fullpath)
        part = sound[0:10000]
        part.export(fullpath,format="wav")
        print(f"{file} split successfully!!!")
        
        
def generate_json():
    
    data = pd.read_csv("rename_data.csv")
    col = data.columns.values
    idx = data['name'].values.tolist()
    for i in idx :
        filename = "Json/" + i + ".json"
        file = open(filename,"w")
        row = idx.index(i)
        dict = {
            "name":data.at[row,col[0]],
            "keywords":data.at[row,col[1]],
            "caption":data.at[row,col[2]],
        }
        json.dump(dict,file,indent=4)
        file.close()
    print("Generation complete!")

def split_dataset():
    
    data = pd.read_csv('rename_data.csv')
    idx = data['name'].values.tolist()
    random.shuffle(idx)
    train = idx[:3500]
    val = idx[3500:4335]
    
    for i in train :
        
        sound = i 
        src = 'music/' + sound
        dest = 'Musics/train/' + sound
        os.replace(src,dest)
        
        json_file = i + '.json'
        src = 'Json/' + json_file
        dest = 'Json/train/' + json_file
        os.replace(src,dest)
        
    for i in val:
        
        sound = i 
        src = 'music/' + sound
        dest = 'Musics/validation/' + sound
        os.replace(src,dest)
        
        json_file = i + '.json'
        src = 'Json/' + json_file
        dest = 'Json/validation/' + json_file
        os.replace(src,dest)
        

if __name__ == '__main__':
   split_dataset()
    