
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, ReLU, Layer, Concatenate

def save_model(model, save_dir, name, epoch):
    model_json = model.to_json()
    with open(save_dir + name + "_epoch_{0}.json".format(epoch), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_dir + name + "_epoch_{0}.h5".format(epoch))
    
def load_model(json_path, weights_path, custom_objects=None):
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    if custom_objects:
      loaded_model = model_from_json(loaded_model_json, custom_objects)
    else:
      loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    return loaded_model