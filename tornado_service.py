# -*- coding: utf-8 -*-
import tornado
from tornado.web import url, RequestHandler
from src.subject_classifier_trt import ClassifierPipeLine
import json
from logger.logger import GetLogger

# classifier_log = GetLogger(log_name="subject_classifier.log", logs_dir='/home/logs/xeslog', logs_backup=7, console_out=False).getLogger()
classifier_log = GetLogger(log_name="subject_classifier.log", logs_dir='./logs', logs_backup=7, console_out=False).getLogger()

class SubjectClassifier(RequestHandler):
    def initialize(self, subject_classifier) -> None:
        self.subject_classifier = subject_classifier

    def post(self):
        log_dict = {}
        try:
            log_dict['headers']= self.request.headers._dict
            body = self.request.body.decode('utf-8')
            log_dict['body'] = body
            data_json = json.loads(body)
            
            log_dict['trace_id'] = data_json['trace_id']
            log_dict['输入数据:']= data_json['text']

            subject,score = self.subject_classifier.pipeline(data_json['text'], log_dict)
            self.write({'trace_id':data_json['trace_id'],'status_code':'0','status_msg':'预测成功','predict_result':{'score':score,'subject':subject}})
            classifier_log.info(json.dumps(log_dict))

        except Exception as e:
            log_dict['错误原因:']=str(e)
            classifier_log.info(json.dumps(log_dict))
            try:
                self.write(self.write({'trace_id':data_json['trace_id'],'status_code':'-1','status_msg':str(e),'predict_result':{'score':-1.0,'subject':'erro'}}))
            except:
                self.write({'message_code':'-1','message':'入参错误:{}'.format(body)})
            


if __name__ == "__main__":

    classifer_pipeline = ClassifierPipeLine()

    application_1 = tornado.web.Application(handlers=[(r"/predict", SubjectClassifier, {"subject_classifier": classifer_pipeline}),])
    application_1.listen(1234)
    print('begin listening on port:{}'.format(1234))

    tornado.ioloop.IOLoop.current().start()

    
