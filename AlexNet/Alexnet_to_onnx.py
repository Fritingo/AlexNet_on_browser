import torch

from inference_Alexnet import AlexNet


def main():
  pytorch_model = AlexNet()
  pytorch_model.load_state_dict(torch.load('cifar100_Alexnet.pt'))
  pytorch_model.eval()
  dummy_input = torch.zeros(128*128*4)
  torch.onnx.export(pytorch_model, dummy_input, 'cifar100_Alexnet.onnx', verbose=True)


if __name__ == '__main__':
  main()
