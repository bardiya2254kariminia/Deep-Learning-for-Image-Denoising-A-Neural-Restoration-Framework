from torchvision.transforms import transforms


class Transform_class(object):

    def get_transform():
        t = {
            "cifar100_transform" : transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5) , std=(0.5,0.5,0.5))
            ])
        }
        return t