from transformers import pipeline
import torch


class ClassifierPipeLine():
    def __init__(self):
        self.model_path = 'src/model/chinese-roberta-wwm-ext'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.subj_cls_pipeline = pipeline(task='text-classification', 
                                          model=self.model_path, 
                                          tokenizer=self.model_path, 
                                          max_length=256,
                                          padding='max_length',
                                          truncation=True,
                                          use_fast=True,
                                          device=self.device)
        self.subject_dic = {'数学':'math', '物理':'physics', '化学':'chemistry', '生物':'bio', '地理':'geography', '历史':'history', '政治':'politics','语文':'chinese','英语':'english','其他': 'other'}
    def pipeline(self, text, log_dict):
        try:
            if len(text)>500:
                text = text[:500]
            res = self.subj_cls_pipeline(text)[0]
            return self.subject_dic[res['label']], res['score']
        except Exception as e:
            log_dict['模型报错'] = e
            raise e