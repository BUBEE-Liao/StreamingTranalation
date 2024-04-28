import torch
import os
import librosa
import numpy as np
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoFeatureExtractor

class TextAudioLoader(torch.utils.data.Dataset):
    def __init__(self, trainList_path):
        super(TextAudioLoader, self).__init__()
        self.train_txt_file_path = trainList_path
        self.train_list = self.get_train_list(self.train_txt_file_path)
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    def get_train_list(self, train_txt_file_path):
        train_list = []
        with open(train_txt_file_path, 'r')as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                if(line != ''):
                    data = line.split('|')
                    train_list.append(data)
        return train_list

    def get_text_audio_target(self, data):
        # audio
        audio_path = data[0]
        # sample_rate = None
        # if(audio_path[-4:]=='.mp3'): sample_rate=48000
        # elif('ner-trs' in audio_path or 'ner-trs-pro' in audio_path):sample_rate=16000
        # else: sample_rate=44100


            
        # audio, sr = librosa.load(audio_path, sr=sample_rate)
        # # audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
        # audio = torch.FloatTensor(audio.astype(np.float32))
        
        # # audio = audio.unsqueeze(dim=0)
        # #text
        # # src_txt = data[1]
        tgt_txt = data[2]

        # # src_txt = self.tokenizer(src_txt, add_special_tokens=False, return_tensors="pt").input_ids
        # tgt_txt = self.tokenizer(tgt_txt, add_special_tokens=False, return_tensors="pt").input_ids

        # # src_txt = src_txt.squeeze()
        # try:
        #     _ = len(src_txt)
        # except:
        #     src_txt = src_txt.unsqueeze(dim=0)
        # # tgt_txt = tgt_txt.squeeze(dim=0)

        # bos_token = self.tokenizer(self.tokenizer.pad_token, add_special_tokens=False, return_tensors="pt").input_ids
        # eos_token = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
        
        # decoder_input_ids = torch.cat((bos_token, tgt_txt), -1)
        # decoder_input_ids = decoder_input_ids.squeeze()

        # target_ids = torch.cat((tgt_txt, eos_token), -1)
        # target_ids = target_ids.squeeze()
        
        # return audio, decoder_input_ids, target_ids
        return audio_path, tgt_txt
        
    
    def __getitem__(self, index):
        return self.get_text_audio_target(self.train_list[index])

    def __len__(self):
        return len(self.train_list)

class TextAudioCollate():

    def __init__(self, return_ids=False):
        self.return_ids = return_ids
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    def __call__(self, batch):

        path = [f[0] for f in batch] # audio path
        tgt_txt = [f[1] for f in batch] # en

        # audio
        audio = [self.load_wav(audio_path) for audio_path in path]
        audio_feature = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding='longest')
        audio_feature_padded = audio_feature['input_features']
        audio_feature_mask = audio_feature['attention_mask']
        # tgt text
        decoder_input_ids, decoder_input_mask, label, tgt_len, label_padding_mask = self.generate_decoder_input_and_label(tgt_txt)
        return {'speech_input':audio_feature_padded, 'speech_attention_mask':audio_feature_mask, 'decoder_input_ids':decoder_input_ids, 'decoder_attention_mask':decoder_input_mask, 'label':label, 'tgt_len':tgt_len, 'label_padding_mask':label_padding_mask}

    def load_wav(self, audio_path):
        sample_rate = None
        if(audio_path[-4:]=='.mp3'): sample_rate=48000
        elif('ner-trs' in audio_path or 'ner-trs-pro' in audio_path):sample_rate=16000
        else: sample_rate=44100
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        if(sample_rate!=16000):
            audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
        return audio

    def generate_decoder_input_and_label(self, tgt_txt):
        batch_size = len(tgt_txt)
        bos_token = self.tokenizer(self.tokenizer.pad_token, add_special_tokens=False, return_tensors="pt").input_ids
        eos_token = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
    
        tgt_ids = [self.tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids for txt in tgt_txt]
        label = [torch.cat((txt_ids, eos_token), -1).squeeze() for txt_ids in tgt_ids]
        decoder_input = [torch.cat((bos_token, txt_ids), -1).squeeze() for txt_ids in tgt_ids]
    
        max_decoder_input_ids_len = max([len(x) for x in decoder_input])
        max_label_len = max([len(x) for x in label])
    
        decoder_input_ids_padded = torch.IntTensor(batch_size, max_decoder_input_ids_len)
        label_padded = torch.IntTensor(batch_size, max_label_len)
        decoder_input_mask = torch.IntTensor(batch_size, max_decoder_input_ids_len)
        label_len = torch.FloatTensor(batch_size)
        label_padding_mask = torch.FloatTensor(batch_size, max_label_len)
    
        decoder_input_ids_padded.fill_(65000)
        label_padded.fill_(65000)
        decoder_input_mask.zero_()
        label_len.zero_()
        label_padding_mask.zero_()
        
        for i in range(batch_size):
            decoder_input_ids = decoder_input[i]
            decoder_input_ids_padded[i, :decoder_input_ids.size(0)] = decoder_input_ids
    
            decoder_input_mask[i, : decoder_input_ids.size(0)] = torch.ones(decoder_input_ids.size(0))
            
            label_ids = label[i]
            label_padded[i, :label_ids.size(0)] = label_ids
            label_len[i] = label_ids.size(0)
            label_padding_mask[i, :label_ids.size(0)] = torch.ones(label_ids.size(0), dtype=torch.float32)
            
        return decoder_input_ids_padded, decoder_input_mask, label_padded, label_len, label_padding_mask