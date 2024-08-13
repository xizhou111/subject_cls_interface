# -*- coding: utf-8 -*-
import json
import requests
import time
from pprint import pprint
def paper_analysis_client(data=None):
    data = json.dumps(data)
    res = requests.post('http://127.0.0.1:1234/predict', data)
    # res = requests.post('http://xbtk-internal.100tal.com/subject-cls/predict', data)
    res = json.loads(res.text)
    res = res['predict_result']
    return res


if __name__ == '__main__':
    input_content = {
                    "trace_id":"12345",
                    "text":"2.(2021北京一七一中学三模，2)2018年10月23日，世界上最长的跨海大桥--港珠澳大桥正式开通",
                }
    s = time.time()
    analysis_result = paper_analysis_client(input_content)
    print(time.time()-s)
    print(analysis_result)
    