
import os, pathlib, h5py

model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), r"converted_tflite_quantized\keras_Model.h5")

f = h5py.File(model_path, mode="r+")
model_config_string = f.attrs.get("model_config")
print("searching")
if model_config_string.find('"groups": 1,') != -1:
    print("Found!")
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
    model_config_string = f.attrs.get("model_config")
    assert model_config_string.find('"groups": 1,') == -1

f.close()

print("searching")
if model_config_string.find('"groups": 1,') != -1:
    print("Found!")
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
    model_config_string = f.attrs.get("model_config")
    assert model_config_string.find('"groups": 1,') == -1

f.close()