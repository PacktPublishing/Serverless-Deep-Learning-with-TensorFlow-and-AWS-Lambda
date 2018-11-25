import boto3
import numpy as np
import tensorflow as tf
import os.path
import re
from urllib.request import urlretrieve
import json

SESSION = None
strBucket = 'serverlessdeeplearning'

def map(event, context):
    dictMap = {}
    dictMap['branch1'] = {'url':event[0]}
    dictMap['branch2'] = {'url':event[1]}
    dictMap['branch3'] = {'url':event[2]}
    return dictMap

def reduce(event, context):
    vecRes = []
    for res in event:
        vecRes.append(res['res'])
    return vecRes

def handler(event, context):
    global strBucket
    global SESSION

    if not os.path.exists('/tmp/imagenet/'):
        os.makedirs('/tmp/imagenet/')

    if SESSION is None:
        downloadFromS3(strBucket,'imagenet/imagenet_2012_challenge_label_map_proto.pbtxt','/tmp/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt')
        downloadFromS3(strBucket,'imagenet/imagenet_synset_to_human_label_map.txt','/tmp/imagenet/imagenet_synset_to_human_label_map.txt')
    
    strFile = '/tmp/imagenet/inputimage.jpg'

    urlretrieve(event, strFile)
    strResult = run_inference_on_image(strFile)

    return strResult

def run_inference_on_image(image):
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    global SESSION
    if SESSION is None:
        SESSION = tf.InteractiveSession()
        create_graph()

    softmax_tensor = tf.get_default_graph().get_tensor_by_name('softmax:0')
    predictions = SESSION.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-5:][::-1]

    node_lookup = NodeLookup()
    strResult = '%s (score = %.5f)' % (node_lookup.id_to_string(top_k[0]), predictions[top_k[0]])
    vecStr = []
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        vecStr.append('%s (score = %.5f)' % (human_string, score))
    return vecStr

def downloadFromS3(strBucket,strKey,strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)

def getObject(strBucket,strKey):
    s3_client = boto3.client('s3')
    s3_response_object = s3_client.get_object(Bucket=strBucket, Key=strKey)
    return s3_response_object['Body'].read()  

def create_graph():
    global strBucket
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(getObject(strBucket,'imagenet/classify_image_graph_def.pb'))
    _ = tf.import_graph_def(graph_def, name='')

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                '/tmp/imagenet/', 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                '/tmp/imagenet/', 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
