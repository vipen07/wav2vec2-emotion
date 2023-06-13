import torch

import torch.nn as nn

import torch.nn.functional as F

import torchaudio


from models import Wav2Vec2ForSpeechClassification

from transformers import AutoConfig, Wav2Vec2FeatureExtractor ,AutoTokenizer





model_name_or_path = "Models/Finetuned"




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(model_name_or_path)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)

sampling_rate = feature_extractor.sampling_rate




# for wav2vec

model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)






def speech_file_to_array_fn(path, sampling_rate):

    speech_array, _sampling_rate = torchaudio.load(path)

    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)

    speech = resampler(speech_array).squeeze().numpy()

    return speech





def predict(path, sampling_rate):

    speech = speech_file_to_array_fn(path, sampling_rate)

    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    inputs = {key: inputs[key].to(device) for key in inputs}




    with torch.no_grad():

        logits = model(**inputs).logits




    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in

               enumerate(scores)]

    return outputs





path = "/data/ganji_sreeram/Interns/Vipendra:Emotion_Recognition/Wav2vec2_Emotion/Data/Emotional_Database/happiness/h01 (1).wav"

outputs = predict(path, sampling_rate)  

print(outputs)



