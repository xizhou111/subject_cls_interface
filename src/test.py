import os
import pandas as pd
import json
import requests
from pprint import pprint
import uuid
from tqdm import tqdm


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report



def get_result(data=None):
    data = json.dumps(data)  
    try:
        res = requests.post('http://xbtk.100tal.com/subject-cls/predict', data, timeout=1)
        # res = requests.post('http://127.0.0.1:1023/subject_classifier', data, timeout=1)
        res = json.loads(res.text)
        return res
    except:
        return False

# 定义评估指标，包括准确率、精确率、召回率、F1值
def compute_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    classification_rep = classification_report(labels, predictions, output_dict=True, digits=4)
    results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    # 添加每个类别的recall指标到结果字典
    for label, metrics in classification_rep.items():
        if isinstance(metrics, dict):
            # results[f'recall_{label}'] = metrics['recall']
            try:
                results[f'recall_{id2label[int(label)]}'] = metrics['recall']
            except:
                results['label'] = metrics['recall']
    return results

if __name__ == '__main__':

    # Load the evaluation dataset
    eval_data_file = '/mnt/cfs/NLP/zcl/subjects_classification/fasttext/eval_data/eval_data.json'
    with open(eval_data_file, 'r') as file:
        eval_data = json.load(file)
    # pprint(eval_data)
    # exit()

    # eval_data = eval_data.rename(columns={"question": "text", "subject_id": "label"})

    label2id = {"other": 0, "chinese": 1, "math": 2, "english": 3, "physics": 4, "chemistry": 5, "bio": 6, "history": 7, "geography": 8, "politics": 9}
    id2label = {0: "其他", 1: "语文", 2: "数学", 3: "英语", 4: "物理", 5: "化学", 6: "生物", 7: "历史", 8: "地理", 9: "政治"}
    
    labels = []
    # for i in range(len(eval_data)):
    #     labels.append(eval_data[i]['subject_id'])

    preds = []
    for i in tqdm(range(len(eval_data))):
        data = {
            "text": eval_data[i]['question'],
            "trace_id": str(uuid.uuid1())
        }
        res = get_result(data)
        if isinstance(res, bool):
            print("Error: get_result returned False")
            continue
        # pprint(res['predict_result']['subject'])
        # exit()
        preds.append(label2id[res['predict_result']['subject']])
        labels.append(eval_data[i]['subject_id'])

    results = compute_metrics(preds, labels)
    pprint(results)




 









