
import tensorflow as tf
import io

from object_detection.utils import dataset_util

def create_tf_example(example):
    '''
        examples: Dict -> {
            'img': np.array,
            'h', 'w',
            'xmins', 'ymins', 'xmax', 'ymax',
            'class_to_id': map[class -> id],
        }
    '''
    # TODO(user): Populate the following variables from your example.
    height = example['h'] # Image height
    width = example['w'] # Image width
    filename = b'' # Filename of the image. Empty if image is not from file
    encoded_image_data = example['img'].tobytes() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'

    xmins = example['xmins'] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = example['xmaxs'] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = example['ymins'] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = example['ymaxs'] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = example['classes_text'] # List of string class name of bounding box (1 per box)
    classes = example['classes'] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example