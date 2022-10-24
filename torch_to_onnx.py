import torch
from models.network_def.iresnet import iresnet50
from torchsummary import summary

# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
device = 'cpu'

model_path = "models/face_recognition_iresnet/backbone.pth"

weight = torch.load(model_path, map_location=device)
model = iresnet50(False)
model.load_state_dict(weight)
model.to(device)
model.eval()


model_path_onnx = "./assets/face_reg/backbone.onnx"

dummy_input = torch.randn(1, 3, 112, 112, requires_grad=True)
inputs = ['input']
outputs = ['output']
dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

torch.onnx.export(model, dummy_input, model_path_onnx, verbose=False, input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes)


print("-------------")