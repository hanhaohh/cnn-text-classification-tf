# -*- coding: utf-8 -*-

import numpy as np
import json
import re
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2


TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                      re.UNICODE)

class TFClient(object):
    """
    TFClient class to use for RPC calls
    """
    def __init__(self, host, port, w2i_table_path):
        self.host = host
        self.port = port
        self.words2index_table = json.load(open(w2i_table_path))
        # Setup channel
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    def group_words(self, text):
        regex = []

        # Match a whole word:
        regex += [ur'\w+']

        # Match a single CJK character:
        regex += [ur'[\u4e00-\ufaff]']

        # Match one of anything else, except for spaces:
        regex += [ur'[^\s]']

        regex = "|".join(regex)
        r = re.compile(regex)
        return r.findall(text)

    def tokenizer(self, iterator):
        for value in iterator:
            yield TOKENIZER_RE.findall(" ".join(self.group_words(value)))

    def transform(self, raw_documents, unused_y=None):
        return_list = []
        for tokens in self.tokenizer(raw_documents):
          word_ids = np.zeros(80, np.int32)
          for idx, token in enumerate(list(tokens)):
              if idx >= 80:
                  break
              word_ids[idx] = self.words2index_table.get(token, 0)
          return word_ids



    def execute(self, request, timeout=10.0):

        return self.stub.Predict(request, timeout)

    def make_prediction(self, data, timeout=10., model_name='inception', convert_to_dict=True):
        """
        Make a prediction on a buffer full of image data (tested .jpg as of now)
        :param data: Data buffer
        :param name: Name of the model_spec to use
        :param timeout: Timeout in seconds to wait for more batches to pile up
        :return: Prediction result
        """
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.version.value = 4
        proto = tf.contrib.util.make_tensor_proto(self.transform(data), shape=[1, 80])
        proto1 = tf.contrib.util.make_tensor_proto(1.0)
        request.model_spec.signature_name = 'sentiment_prediction'
        request.inputs["inputs"].CopyFrom(proto)
        request.inputs["dropout_rate"].CopyFrom(proto1)
        response = self.execute(request, timeout=timeout)

        if not convert_to_dict:
            return response

        # Convert to friendly python object
        results_dict = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results_dict[key] = nd_array
        return results_dict

client = TFClient("localhost", 9000, "result.json")
import time
a = time.time()
print client.make_prediction(data=[u"这个东西太垃圾饿了"], model_name="inception")
b = time.time()
print b-a
