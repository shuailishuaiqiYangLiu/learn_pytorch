import torchvision

# train_data = torchvision.datasets.ImageNet("./dataset_imagenet",split='train',
#                                            transform=torchvision.transforms.ToTensor())

# vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)