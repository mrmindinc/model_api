import numpy as np
from comparison import cosine_similarity_check
from operator import itemgetter
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
koo_sbert_model = torch.load("MODEL/koo_sbert_model", map_location=device)
print('device :', device)

class ExtractIntent():
      
    def nlp_process(self, data):

        _data = data['data']
        
        in_emb_text = koo_sbert_model.encode(data['input_text'], convert_to_tensor=True)

        for info in _data:
            comp_text = koo_sbert_model.encode(info['text'], convert_to_tensor=True)
            confidence = cosine_similarity_check(in_emb_text, comp_text).tolist()[0] * 100
            info['cosine_similarity'] = confidence

        data_fin = sorted(_data, key=itemgetter('cosine_similarity'), reverse=True)

        return data_fin[0]


    def nlp_process_sculpture(self, data):

        _data = data['data']
        
        in_emb_text = koo_sbert_model.encode(data['input_text'], convert_to_tensor=True)

        for info in _data:
            comp_text = koo_sbert_model.encode(info['input'], convert_to_tensor=True)
            confidence = cosine_similarity_check(in_emb_text, comp_text).tolist()[0] * 100
            info['cosine_similarity'] = confidence

        data_fin = sorted(_data, key=itemgetter('cosine_similarity'), reverse=True)

        return data_fin[0]


