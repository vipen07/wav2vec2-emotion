import torch

import torch.nn as nn

import torch.nn.functional as F

import torchaudio
<<<<<<< HEAD
from models import Wav2Vec2ClassificationHead, Wav2Vec2ForSpeechClassification
from transformers import AutoConfig, Wav2Vec2Processor ,AutoTokenizer



from transformers import AutoConfig, Wav2Vec2FeatureExtractor

import librosa

import IPython.display as ipd

import numpy as np

import pandas as pd

 

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_name_or_path = "Models/Finetuned"



config = AutoConfig.from_pretrained(model_name_or_path)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)

sampling_rate = feature_extractor.sampling_rate

model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

 
=======
from transformers import AutoConfig, Wav2Vec2FeatureExtractor


model_name_or_path = "/models/<model_name>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

# for wav2vec
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# for hubert


>>>>>>> fd4315da0389e49ffaa7ad128e49ed14ab92450d

def speech_file_to_array_fn(path, sampling_rate):

    speech_array, _sampling_rate = torchaudio.load(path)
<<<<<<< HEAD

    resampler = torchaudio.transforms.Resample(_sampling_rate)

=======
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
>>>>>>> fd4315da0389e49ffaa7ad128e49ed14ab92450d
    speech = resampler(speech_array).squeeze().numpy()

    return speech



processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)

def predict(path, sampling_rate):

    speech = speech_file_to_array_fn(path, sampling_rate)
<<<<<<< HEAD

    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)



    input_values = features.input_values.to(device)

    attention_mask = features.attention_mask.to(device)
=======
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}
>>>>>>> fd4315da0389e49ffaa7ad128e49ed14ab92450d



    with torch.no_grad():
<<<<<<< HEAD

        logits = model(input_values, attention_mask=attention_mask).logits
=======
        logits = model(**inputs).logits
>>>>>>> fd4315da0389e49ffaa7ad128e49ed14ab92450d



    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
<<<<<<< HEAD

    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]

    return outputs




STYLES = """

<style>

div.display_data {

    margin: 0 auto;

    max-width: 500px;

}

table.xxx {

    margin: 50px !important;

    float: right !important;

    clear: both !important;

}

table.xxx td {

    min-width: 300px !important;

    text-align: center !important;

}

</style>

""".strip()



def prediction(df_row):

    path, emotion = df_row["path"], df_row["emotion"]

    df = pd.DataFrame([{"Emotion": emotion, "Sentence": "    "}])

    setup = {

        'border': 2,

        'show_dimensions': True,

        'justify': 'center',

        'classes': 'xxx',

        'escape': False,

    }

    ipd.display(ipd.HTML(STYLES + df.to_html(**setup) + "<br />"))

    speech, sr = torchaudio.load(path)

    speech = speech[0].numpy().squeeze()

    speech = librosa.resample(np.asarray(speech), orig_sr = sr, target_sr=sampling_rate)

    ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=sampling_rate))



    outputs = predict(path, sampling_rate)

    r = pd.DataFrame(outputs)

    ipd.display(ipd.HTML(STYLES + r.to_html(**setup) + "<br />"))

test = pd.read_csv("Data/test.csv", sep="\t")

test.head()

print(prediction(test.iloc[0]))
prediction(test.iloc[1])
prediction(test.iloc[2])


=======
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs


path = "/content/data/aesdd/disgust/d01 (2).wav"
outputs = predict(path, sampling_rate)    
>>>>>>> fd4315da0389e49ffaa7ad128e49ed14ab92450d
