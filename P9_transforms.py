from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# tensor数据类型
# 通过transforms.ToTensor看两个问题
# 1.transforms该如何使用
# 2.为什么需要Tensor数据类型

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter('logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()