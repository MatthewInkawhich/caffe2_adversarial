# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package lmdb_create_example
# Module caffe2.python.examples.lmdb_create_example
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import random
import numpy as np
import lmdb
from scipy.misc import imread
import cv2
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper

# handle command line arguments
parser = argparse.ArgumentParser(description='Converts a directory of images to an LMDB using a label file')
parser.add_argument('-l', '--labels', help='path to labels file', required=True)
parser.add_argument('-o', '--output', help='name of output lmdb', required=True)
parser.add_argument('-s', '--shuffle', action='store_true', help='if set, data is shuffled before going conversion', required=False)
args = vars(parser.parse_args())


# Read labels file into list (for shuffling purposes)
with open(args['labels']) as f:
    content = f.readlines()
content = [x.rstrip() for x in content]
if (args['shuffle']):
    random.shuffle(content)
print(content)


print(">>> Write database...")
LMDB_MAP_SIZE = 1 << 40   # MODIFY: just a very large number
print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)
env = lmdb.open(args['output'], map_size=LMDB_MAP_SIZE)


with env.begin(write=True) as txn:
    count = 0
    for line in content:
        img_file = line.split()[0]
        label = int(line.split()[1])
        print(img_file, label)
        img_data = imread(img_file)

        # ensure that 1 channel images get channel dimension (1)
        if (img_data.ndim < 3):
           img_data = np.expand_dims(img_data, axis=0)

        # Create TensorProtos
        tensor_protos = caffe2_pb2.TensorProtos()
        img_tensor = tensor_protos.protos.add()
        img_tensor.dims.extend(img_data.shape)
        img_tensor.data_type = 1

        flatten_img = img_data.reshape(np.prod(img_data.shape))
        img_tensor.float_data.extend(flatten_img)

        label_tensor = tensor_protos.protos.add()
        label_tensor.data_type = 2
        label_tensor.int32_data.append(label)
        txn.put(
            '{}'.format(count).encode('ascii'),
            tensor_protos.SerializeToString()
        )

        if ((count % 100 == 0)):
            print("Inserted {} rows".format(count))
        count = count + 1

print("Inserted {} rows".format(count))
