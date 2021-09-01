import os
import glob
import argparse
import matplotlib
import torch

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
# from keras.models import load_model
# from tensorflow.python.keras.models import load_model
from model_dense161 import Model
# from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='densedepth_161', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
# custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
if args.model == "my_densenet_169":
    from model_dense169 import Model
    model = Model().cuda()
else:
    model = Model().cuda()
# modelload_model(args.model, custom_objects=custom_objects, compile=False)
model.load_state_dict(torch.load('%s.pth'%(args.model)))

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images(glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

model.eval()

# Compute results
outputs = predict(model, inputs)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,6))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
