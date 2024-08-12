import numpy as np
from transformers import AutoTokenizer
import tensorrt as trt
from src import common


TRT_LOGGER = trt.Logger(trt.ILogger.ERROR)

class ClassifierPipeLine():
    def __init__(self):
        self.engine_model_path = "model/roberta_pretrain_512.trt"
        self.onnx_model_path = "model/roberta_pretrain_512_static.onnx"
        self.batch_size = 1
        self.max_sequence_length = 256
        # self.engine = self.get_engine(self.engine_model_path)
        self.engine_model_path = self.convert_onnx_to_engine(self.onnx_model_path, self.engine_model_path, max_batch_size=self.batch_size)
        self.engine = self.get_engine(self.engine_model_path)
        self.context = self.get_context(self.engine)

        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers_v2(self.engine, self.context)

        self.transformers_model_path = "trt_model/chinese-roberta-wwm-ext-tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformers_model_path, use_fast=True)
        
        self.id2label = {0: "other", 1: "chinese", 2: "math", 3: "english", 4: "physics", 5: "chemistry", 6: "bio", 7: "history", 8: "geography", 9: "politics"}

    def get_engine(self, engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine
    
    def get_context(self, engine):
        context = engine.create_execution_context()
        context.active_optimization_profile = 0
        origin_inputshape = context.get_binding_shape(0)                # (1,-1) 
        # (batch_size, max_sequence_length)
        origin_inputshape[0], origin_inputshape[1] = self.batch_size, self.max_sequence_length
        context.set_binding_shape(0, (origin_inputshape))               
        context.set_binding_shape(1, (origin_inputshape))
        return context
    
    def convert_onnx_to_engine(self, onnx_filename,
                           engine_filename = None,
                           max_batch_size = 1,
                           max_workspace_size = 1 << 30,
                           fp16_mode = False):
        logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(logger) as builder, \
                builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                trt.OnnxParser(network, logger) as parser, \
                builder.create_builder_config() as config:
            
            config.max_workspace_size = max_workspace_size
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            builder.max_batch_size = max_batch_size

            print("Parsing ONNX file.")
            with open(onnx_filename, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            print('Completed parsing of ONNX file')

            print("Building TensorRT engine. This may take a few minutes.")
            engine = builder.build_engine(network, config)
            if engine is None:
                print("Failed to create engine.")
                return None

            if engine_filename:
                with open(engine_filename, 'wb') as f:
                    f.write(engine.serialize())
            print("Created engine success! ")

            return engine_filename

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def softmax(self, x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def trt_inference(self, text):
        input = self.tokenizer.encode_plus(text,  
                                           add_special_tokens=True, 
                                           truncation=True,
                                           max_length=self.max_sequence_length, 
                                           padding="max_length", 
                                           return_attention_mask=True,
                                           return_tensors='pt',)
        
        tokens_id =  self.to_numpy(input['input_ids'].int())
        attention_mask = self.to_numpy(input['attention_mask'].int())

        self.inputs[0].host = tokens_id
        self.inputs[1].host = attention_mask

        trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        preds = np.argmax(trt_outputs, axis=1)
        index = int(preds[0])
        label = self.id2label[index]
        score = self.softmax(trt_outputs[0])[index]
        return label, float(score)

    def pipeline(self, text, log_dict):
        try:
            return self.trt_inference(text)
        except Exception as e:
            log_dict['模型报错'] = e
            raise e
