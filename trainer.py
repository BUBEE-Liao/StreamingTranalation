import torch
import numpy as np
from streamingTranslationModel import StreamingTranslationModel
from transformers import AutoFeatureExtractor, MarianTokenizer
from transformers import AutoProcessor, Wav2Vec2Model
from transformers import Trainer, TrainingArguments
import librosa
import torch.nn as nn
from datasets import load_metric
import random
import json
from loss import true_len, monotonic_alignment, varianceLoss, DAL_loss
from data_loader import TextAudioLoader, TextAudioCollate
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.cuda.amp import GradScaler, autocast

# metric = load_metric("sacrebleu")
config = json.load(open('config.json'))
num_head = config['decoder_attention_heads']
num_decoder_layer = config['decoder_layers']
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
def data_collator(features: list):
    path = [f[0] for f in features] # audio path
    tgt_txt = [f[1] for f in features] # en
    # audio
    audio = [load_wav(audio_path) for audio_path in path]
    audio_feature = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding='longest')
    audio_feature_padded = audio_feature['input_features']
    audio_feature_mask = audio_feature['attention_mask']

    
    # tgt text
    decoder_input_ids, decoder_input_mask, label, tgt_len, label_padding_mask = generate_decoder_input_and_label(tgt_txt)
    return {'speech_input':audio_feature_padded, 'speech_attention_mask':audio_feature_mask, 'decoder_input_ids':decoder_input_ids, 'decoder_attention_mask':decoder_input_mask, 'label':label, 'tgt_len':tgt_len, 'label_padding_mask':label_padding_mask}

def load_wav(audio_path):
    sample_rate = None
    if(audio_path[-4:]=='.mp3'): sample_rate=48000
    elif('ner-trs' in audio_path or 'ner-trs-pro' in audio_path):sample_rate=16000
    else: sample_rate=44100
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if(sample_rate!=16000):
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
    return audio
def generate_decoder_input_and_label(tgt_txt):
    batch_size = len(tgt_txt)
    bos_token = tokenizer(tokenizer.pad_token, add_special_tokens=False, return_tensors="pt").input_ids
    eos_token = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids

    tgt_ids = [tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids for txt in tgt_txt]
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
    
train_data = []
eval_data = []
# with open('finalDataset.txt', 'r')as f:
with open('finalDataset.txt', 'r')as f:
    for line in f.readlines():
        if(line!=''):
            data = line.split('|') # path, zh, en
            train_data.append(
            [data[0],
             data[2].rstrip('\n')
            ]
        )
        
with open('eval.txt', 'r')as f:
    for line in f.readlines():
        if(line!=''):
            data = line.split('|') # path, zh, en
            eval_data.append(
            [data[0],
             data[2].rstrip('\n')
            ]
        )

def compute_metrics(eval_pred):
    prediction, label = eval_pred
    speech_logits, alpha, src_len, tgt_len, label_padding_mask = prediction
    speech_logits = torch.from_numpy(speech_logits)
    alpha = torch.from_numpy(alpha)
    src_len = torch.from_numpy(src_len)
    tgt_len = torch.from_numpy(tgt_len)
    label_padding_mask = torch.from_numpy(label_padding_mask)
    label = torch.from_numpy(label)
    
    logSoftmax = nn.LogSoftmax(dim=-1)
    nllloss = nn.NLLLoss(reduction='mean', ignore_index=-100)
    speech_logits_logSoftmax = logSoftmax(speech_logits)
    speech_logits_logSoftmax = speech_logits_logSoftmax.permute(0, 2, 1)
    label = label.type(torch.LongTensor)
    loss_speech = nllloss(speech_logits_logSoftmax, label)
    
    batch_size = 2
    predictions_txt = tokenizer.batch_decode(torch.argmax(speech_logits, dim=-1),skip_special_tokens=True)
    labels_txt = tokenizer.batch_decode(label,skip_special_tokens=True)
    idx = random.randint(0,batch_size-1)
    print('prediction:',predictions_txt[idx])
    print('label:',labels_txt[idx])

    Dal_loss = 0.4*DAL_loss(alpha, src_len, tgt_len, label_padding_mask, num_head, num_decoder_layer)
    # Variance_loss = 0.001*varianceLoss(alpha)
    # result = metric.compute(predictions=predictions_txt, references=[[label] for label in labels_txt])
    return {
        'loss_speech ': loss_speech.item(),
        'Dal_loss ': Dal_loss.detach()
        # 'Variance_loss ': Variance_loss.detach()
    }

class MyTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False): ## define forward batch data
      label = inputs.get("label")
      tgt_len = inputs.get("tgt_len")
      label_padding_mask = inputs.get("label_padding_mask")

      speech_logits, speech_encoder_attention_mask, p_choose = model(inputs.get('speech_input'), inputs.get('speech_attention_mask'), inputs.get('decoder_input_ids'), inputs.get('decoder_attention_mask')) 
                    
      logSoftmax = nn.LogSoftmax(dim=-1)
      nllloss = nn.NLLLoss(reduction='mean', ignore_index=-100)
      
      # loss -log(Y|X)
      speech_logits_logSoftmax = logSoftmax(speech_logits)
      speech_logits_logSoftmax = speech_logits_logSoftmax.permute(0, 2, 1)
      label = label.type(torch.LongTensor).to(torch.device('cuda'))
      loss_speech = nllloss(speech_logits_logSoftmax, label)#.to(torch.float16)
      # print('loss_speech:', loss_speech.item())

      alpha = monotonic_alignment(p_choose)
      # if(torch.isnan(alpha).any()):
      #     print('==============================nan detect==============================')
      src_len = torch.FloatTensor([true_len(t) for t in speech_encoder_attention_mask])
      # loss DAL
      Dal_loss = DAL_loss(alpha, src_len, tgt_len, label_padding_mask, num_head, num_decoder_layer)
      # with open('outputLOG.txt', 'a')as f:
      #     print('Dal_loss:', 0.1*Dal_loss, file=f)
      # loss Variance
      # Variance_loss = varianceLoss(alpha)
      # print('Variance_loss:', 0.001*Variance_loss)
      # with open('outputLOG.txt', 'a')as f:
      #     print('Variance_loss:\n', 0.1*Variance_loss, file=f)
      # total_loss = loss_speech + (0.4*Variance_loss).to(torch.float16) + (0.4*Dal_loss).to(torch.float16)
      # total_loss = loss_speech + (0.001*Variance_loss) + (0.1*Dal_loss)
      total_loss = loss_speech + (0.4*Dal_loss)
      return (total_loss, {'outputs':speech_logits}) if return_outputs else total_loss

  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
      label = inputs.get("label")
      tgt_len = inputs.get("tgt_len")
      label_padding_mask = inputs.get("label_padding_mask")
      # model.text_decoder.training = False
        
      speech_logits, speech_encoder_attention_mask, p_choose = model(inputs.get('speech_input'), inputs.get('speech_attention_mask'), inputs.get('decoder_input_ids'), inputs.get('decoder_attention_mask')) 
      alpha = monotonic_alignment(p_choose)
      src_len = torch.FloatTensor([true_len(t) for t in speech_encoder_attention_mask])
      return (None, (speech_logits.detach(), alpha.detach(), src_len.detach(), tgt_len.detach(), label_padding_mask.detach()), label)


  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
      model.train()
      for param in model.speech_encoder.parameters():
          param.requires_grad = False 
      # for param in model.text_decoder.embed_tokens.parameters():
      #     param.requires_grad = False 
      # scaler = GradScaler()
      inputs = self._prepare_inputs(inputs)
      loss = self.compute_loss(model, inputs)
      loss = loss / self.args.gradient_accumulation_steps
      # loss = scaler.scale(loss)
      loss.backward()  
      # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
      return loss.detach()
      
      # if is_sagemaker_mp_enabled():
      #   loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
      #     return loss_mb.reduce_mean().detach().to(self.args.device)
          
      # with self.compute_loss_context_manager():
      #     loss = self.compute_loss(model, inputs)

      # if self.args.n_gpu > 1:
      #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

      # if self.use_apex:
      #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
      #         scaled_loss.backward()

      # else:
      #     self.accelerator.backward(loss)

      # return loss.detach() / self.args.gradient_accumulation_steps

    
  # def get_train_dataloader(self):
  #     batch_size=16
  #     train_dataset = TextAudioLoader('finalDataset.txt')
  #     collate_fn = TextAudioCollate()
  #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=8)
  #     return train_loader
    
      

### Define Hyperparameters ###
model = StreamingTranslationModel()
for param in model.speech_encoder.parameters():
        param.requires_grad = False 
model.text_decoder.training = True
batch_size = 4
learning_rate = 1e-4
epochs = 5
result_dir = './ckpt3'
##############################
training_args = TrainingArguments(output_dir=result_dir,
                         do_train=True,
                         do_eval=True,
                         evaluation_strategy='steps',
                         eval_steps=200,
                         prediction_loss_only=False,
                         per_device_train_batch_size=batch_size,
                         per_device_eval_batch_size=batch_size,
                         gradient_accumulation_steps=8,
                         learning_rate=learning_rate,
                         weight_decay=1e-6,
                         adam_beta1=0.9,
                         adam_beta2=0.98,
                         num_train_epochs=epochs,
                         save_strategy="steps",
                         save_steps=7000,
                         save_total_limit=10,
                         # fp16=True,
                         # half_precision_backend ="auto",
                         dataloader_num_workers=4,
                         dataloader_drop_last=False,    
                         logging_dir='./log3',
                         logging_strategy='steps',
                         logging_steps=400,
                         # max_steps=28880
                         )

trainer = MyTrainer(
        model=model, 
        args=training_args,  # training arguments, defined above
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics = compute_metrics
    )
# trainer.train(resume_from_checkpoint=True)
trainer.train(resume_from_checkpoint=False)
trainer.save_model(result_dir)