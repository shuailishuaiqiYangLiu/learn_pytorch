from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

wirter = SummaryWriter("logs")
img = Image.open("images/pytorch.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
wirter.add_image("ToTensor",img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
wirter.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
wirter.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
wirter.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    wirter.add_image("RandomCrop", img_crop, i)

wirter.close()