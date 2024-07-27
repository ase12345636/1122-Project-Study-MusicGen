# 1122-Project-Study-MusicGen

## 主題：音樂生成 Music Generation

## 成員：
- 110321015 陳奕羱
- 110321018 張簡雲翔
- 110321064 劉德權

## 指導老師：楊峻權教授

## Abstract
在現在生成式AI蓬勃發展的年代，無論是在文字或者影像的生成式AI都有卓越表現的情況下，慢慢也開展對於音樂生成式AI的研究，並分成Symbolic跟Audio的不同資料類型進行研究與生成。透過了解到Transformer以及EnCodec的原理以及架構後，進而推展並組成MusicGen的模型，去實作一個音樂生成的AI模型。使用MusicCap資料集作為訓練以及測試資料集，並成功實作出一個MusicGen的模型架構，在參數量約1億左右的情況下得到FAD 10.82677的成績。

## Dataset
### Introduction
MusicCap資料集包含5521首音樂，檔案格式均為無壓縮的.wav檔，sampling rate為48kHz，雙聲道，曲風包羅萬象，從柔和抒情到搖滾重金屬均有，內容部分有些為包含人聲之內容，亦有單純只有音樂內容，資料集內容主要收集YouTube上的影音資料。這份資料集被用在MusicLM中當作training set以及test set，並在MusicGen中當作test  set之一。每首音樂均有相對應的英文關鍵字，如「pop, tinny wide hi hats, mellow piano melody, high pitched female vocal melody, sustained pulsating synth lead」及相對應的caption，如「A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass」，透過上述兩點便可用來形容這首音樂的特徵。因為MusicGen所使用的Dataset皆需要付費，為了成本考量，選用本份資料集作為training set以及test set。

### Data spilt
在下載下來的資料集中，只有5419筆資料，所以就用這5419筆資料作為我們資料集。藉由資料有無經過過濾，分成兩個版本，第一版就是沒經過任何過濾，所有Dataset中的資料都拿來使用的，先將所有的資料進行打亂後，其中先切出80%的資料作為training set，剩下的20%作為test set，在training set中再切出20%作為validation set，換言之，最後的資料集分布，所有資料的65%作為training set、15%作為validation set、20%作為test set。
第二版資料集是有經過人工篩選後的資料集，將第　一個版本的training set、validation set、test set，逐一篩選掉音檔中有出現人聲、雜訊或者明顯不是音樂的內容，經果篩選後，training set中有1343筆資料、validation set中有302筆資料、test set中有360筆資料。

### Data Preprocess
Data Preprocessing分為兩個部分，一個text description，使用dataset的caption；另一個是audio input，使用dataset的音樂。由於T5 Encoder與EnCodec的參數是fixed，為了減少訓練時間，所以本次實驗將text description轉成condition token，audio input轉成audio token。本次模型採用classifier free guidance，所以在text description上，會先根據word drop rate隨機drop一些text（圖十二），取前250個text，不足的padding 0（Ls = 250），並輸入進text encoder產生condition token， word drop rate為0.2。Audio input的部分則只取前5秒並從雙聲道轉成單聲道，sample降至32kHz（La = 5*32000）並透過EnCodec轉成audio token。因為本次實驗以delay pattern進行訓練和生成，所以以delay pattern的形式儲存以便後續訓練。training set的audio token會被當成ground truth用於訓練中計算loss，validation set與test set的audio token則會被用於計算performance metrics。
對training set、validation set、test set資料集獨立處理，為避免處理及訓練過程中記憶體使用量過大，所以採分批處理的方式：每次處理30筆資料並以1個batch的形式儲存。

## Instruction
### Requriments
請參考requirements.txt。

### Hyperparameter
透過調整Config中的檔案，可以調整模型的hyperparameter，預先有調整好100M、300M、800M的參數量的模型。

### Train
如果有改過Config的檔案，需要先調整Train_MusicGen_*.py上方的import，之後執行該檔案，之後開始訓練模型，會將模型儲存到ModelSave資料夾中，如有需要，可以調整.py中180~279行的檔案名稱，避免洗掉以前存的模型，會存loss最佳跟最後一個epoch訓練出來的模型，並在訓練完後產生learning curve。

### Infernece
先改Inference_MusicGen.py中第35行中，變更要載入進來infernece的模型，並執行該檔案，就會開始透過vaildation/test set所提供描述的音樂，之後儲存在Dateset/InferenceData中。

## Report
https://drive.google.com/file/d/1WqE74S8U5OBAGjju2AKNqF1okeLAek57/view?usp=drive_link
