import sys
import torch
from PIL import Image
from recognize_anything.ram.models import ram_plus
from recognize_anything.ram import get_transform


class RAM:
    def __init__(self,
                 device="cuda",
                 ckpt="assets/ram_plus_swin_large_14m.pth"):
        original_stdout = sys.stdout
        sys.stdout = open("nul", "w")
        self.device = device
        self.ckpt = ckpt
        self.image_size = 384
        self.transform = get_transform(image_size=self.image_size)
        with open("./assets/ram_tag_list.txt", "r") as fr:
            tag_list = fr.readlines()
        self.tag_list = [tag.strip().lower() for tag in tag_list]
        delete_tag = ["tattoo artist", "tattoo"]
        delete_tag_index = [self.tag_list.index(tag) for tag in delete_tag]
        model = ram_plus(pretrained=self.ckpt,
                         image_size=self.image_size,
                         vit='swin_l',
                         delete_tag_index=delete_tag_index)
        model.eval()
        self.model = model.to(device)
        sys.stdout = original_stdout

    def process(self, images):
        images = [self.transform(image) for image in images]
        images = torch.stack(images, 0).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate_tag_with_score(images)
        results = []
        for output in outputs:
            result = {"tag_set": output[0], "tag_score": output[1]}
            results.append(result)
        return results




if __name__ == "__main__":
    images = [Image.open("images/demo/demo1.jpg")] * 2

    ram = RAM()
    results = ram.process(images)

    print(results)
