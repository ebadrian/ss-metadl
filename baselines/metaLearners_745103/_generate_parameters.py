import torch
from src.learner.pretrained_encoders.mixup_wrap import resnet152_mix, resnetmix50, mobilenetmix, wide_resnet50_2_mix

print("Generate resnetmix50 parameter")
model = resnetmix50(False).model
torch.save(model.state_dict(), './resources/resnet50.mdl')

print("Generate resnetmix152 parameter")
model = resnet152_mix(False).model
torch.save(model.state_dict(), './resources/resnet152.mdl')

print("Generate wrn50 parameter")
model = wide_resnet50_2_mix(False).model
torch.save(model.state_dict(), './resources/wrn50.mdl')

print("Generate mobilenet parameter")
model = mobilenetmix(False).model
torch.save(model.state_dict(), './resources/mobilenet.mdl')
