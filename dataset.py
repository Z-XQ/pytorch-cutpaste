from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


class MVTecAT(Dataset):
    """MVTec anomaly detection dataset.
    Link: https://www.mvtec.com/company/research/datasets/mvtec-ad
    """

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        # find test images
        if self.mode == "train":
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend((self.root_dir / self.defect_name / "train" / "good").glob(ext))
            self.image_names = list(image_paths)
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            print("loading train images", len(self.image_names))
            # during training we cache the smaller images for performance reasons (not a good coding style)
            self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            print(f"loaded {len(self.imgs)} images")
        else:
            # test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            image_paths = []
            for ext in image_extensions:
                # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / ext)))
                image_paths.extend((self.root_dir / self.defect_name / "test").glob(str(Path("*") / ext)))
            self.image_names = list(image_paths)
            print("loading test images", len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"
