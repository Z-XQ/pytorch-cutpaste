import csv
import os
import shutil
import csv

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import MVTecAT
# from cutpaste import CutPaste
from model import ProjectionNet
import matplotlib.pyplot as plt

from density import GaussianDensitySklearn, GaussianDensityTorch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# test_data_eval = None
# test_transform = None
# cached_type = None

def grad_cam(model, input_tensor, target_layer, target_class=None):
    # 注册hook以获取目标层的输出和梯度
    features = []
    def hook_fn(module, input, output):
        features.append(output)
    handle = target_layer.register_forward_hook(hook_fn)

    # 前向传播得到预测结果
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1

    # 反向传播计算梯度
    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True)

    # 获取特征图与梯度
    feature_map = features[-1]
    gradients = target_layer.weight.grad

    # 计算权重：全局平均池化
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

    # 计算加权求和
    cam = (weights * feature_map).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)  # ReLU激活

    # 归一化到0-1之间
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cam.squeeze().cpu().numpy()

    # 清除hook
    handle.remove()
    return cam


class Classifier:
    def __init__(self, model_path, data_path, defect_type, dist_threshold_val, size=256,
                 head_layer=2,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.head_layer = head_layer
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self._create_model()

        self.data_path = data_path
        self.size = size
        self.defect_type = defect_type
        self.dist_threshold_val = dist_threshold_val
        self.mean = mean
        self.std = std
        self.test_transform = self._get_transform()
        self.trainer_feature_density = None
        self._get_trainer_feature()

        header_list = ["ng num", "ok num", "fn num", "fp num", "fn rate", "fp rate"]

    def _get_trainer_feature(self):
        train_embed = self._get_train_embeds()  # 计算并记载训练集的embed
        train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)  # (image_image_num,512)

        self.trainer_feature_density = GaussianDensityTorch()
        print(f"using density estimation {self.trainer_feature_density.__class__.__name__}")
        self.trainer_feature_density.fit(train_embed)
        # self.trainer_feature_density.mean.to(self.device)
        # self.trainer_feature_density.inv_cov.to(self.device)

    def _create_model(self):
        # create model  加载模型并载入权重
        print(f"loading model {self.model_path}")
        head_layers = [512] * self.head_layer + [128]
        print(head_layers)
        weights = torch.load(self.model_path)
        classes = weights["out.weight"].shape[0]
        self.model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    def _get_transform(self):
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((self.size, self.size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=self.mean,
                                                                 std=self.std))
        # self.test_transform = test_transform
        return test_transform

    def test_dataset(self):
        test_data_eval = MVTecAT(self.data_path, self.defect_type, self.size, transform=self.test_transform,
                                 mode="test")

        dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                     shuffle=False, num_workers=0)

        # get embeddings for test data
        labels = []
        embeds = []
        x_list = []
        with torch.no_grad():
            for x, label in dataloader_test:
                embed, logit = self.model(x.to(self.device))

                # save
                embeds.append(embed.cpu())  # embed(64,512)是resnet18输出的特征
                labels.append(label.cpu())  # label(64,)是标注，0/1
                x_list.append(x)
        labels = torch.cat(labels)  # （image_num,)
        embeds = torch.cat(embeds)  # (image_num,512)
        inputs = torch.cat(x_list)

        # train_embed = self._get_train_embeds()  # 计算并记载训练集的embed

        # norm embeds 测试集，训练集
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)  # (test_image_num,512)
        # train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)  # (image_image_num,512)

        # density = GaussianDensityTorch()
        # print(f"using density estimation {density.__class__.__name__}")
        # density.fit(train_embed)
        distances = self.trainer_feature_density.predict(embeds)  # (image_num,)
        # print(distances)
        # TODO: set threshold on mahalanobis distances and use "real" probabilities

        # 创建图形
        fig, ax = plt.subplots(1)
        colormap = ["b", "r", "c", "y"]  # 传入label的意义就是绘制时，label为1，则绘制点为红色，为0则是蓝色。
        # 绘制散点图
        image_num = distances.shape[0]
        y = np.arange(image_num)
        dir_path = os.path.dirname(self.model_path)
        base_name = os.path.basename(self.model_path)
        save_fig_path = os.path.join("eval", dir_path, base_name)
        os.makedirs(save_fig_path, exist_ok=True)
        ax.scatter(x=distances, y=y, color=[colormap[l] for l in labels])
        plt.xlabel("distance value")
        plt.ylabel("image number")
        fig.savefig(save_fig_path + "/" + self.defect_type +".png")
        plt.close()

        save_ng_dir = os.path.join(data_path+"_res", "ng")
        save_ok_dir = os.path.join(data_path+"_res", "ok")
        distances_array = distances.numpy()
        fp_num = 0
        fn_num = 0
        ng_num = 0
        ok_num = 0
        for i in range(len(distances_array)):
            test_image_path = str(test_data_eval.image_names[i])
            label = os.path.basename(os.path.dirname(test_image_path))
            if label == "good":
                ok_num += 1
            else:
                ng_num += 1
            # image = cv2.imread(test_image_path)
            img = Image.open(test_image_path)
            img = img.resize((self.size, self.size)).convert("RGB")
            image = np.asarray(img)
            print(test_image_path, distances_array[i])
            image_dir_name = os.path.basename(os.path.dirname(test_image_path))
            if self.dist_threshold_val < distances[i]:  # 判为ng
                if image_dir_name == "good":
                    fp_num += 1
                else:
                    pass  # 没有误判
                save_full_path = test_image_path.replace(data_path, save_ng_dir)
                # Grad CAM
                # target_layers = [self.model.resnet18.layer4[-1]]
                target_layers = [self.model.resnet18.layer4]
                cam = GradCAM(model=self.model, target_layers=target_layers)
                # targets = [ClassifierOutputTarget(281)]     # cat
                # targets = [ClassifierOutputTarget(1)]  # dog
                targets = None  # dog

                img_tensor = inputs[i].unsqueeze(0)
                grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                # view
                # img_resized = cv2.resize(image, (self.size, self.size))
                visualization = show_cam_on_image(image.astype(dtype=np.float32) / 255.,
                                                  grayscale_cam, use_rgb=True)
                # img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                output = np.hstack([visualization, image])
                # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            else:  # 判为ok
                if image_dir_name != "good":
                    fn_num += 1
                save_full_path = test_image_path.replace(data_path, save_ok_dir)
                output = image
            save_path_dir = os.path.dirname(save_full_path)
            os.makedirs(save_path_dir, exist_ok=True)
            # shutil.copy(test_image_path, save_full_path)
            outMat = Image.fromarray(output, 'RGB')
            outMat.save(save_full_path)
            # cv2.imwrite(save_full_path, output)

        print(self.defect_type, "误判率: ", fp_num / image_num*1.0, "漏检率：", fn_num / image_num*1.0)
        fp_rate = fp_num / image_num*1.0
        fn_rate = fn_num / image_num*1.0

        return [ok_num, ng_num, fp_num, fn_num, fp_rate, fn_rate]

    # def infer_one_image_path(self, image_path):
    #     print(test_file_path)
    #     image = cv2.imread(test_file_path)
    #     self.infer_one_image(image)

    def infer_one_image(self, image_path):
        # prepare image
        image = cv2.imread(test_file_path)
        img_tensor = self.preprocess_image(image)
        embed, logit = self.model(img_tensor.to(self.device))
        embeds = torch.nn.functional.normalize(embed, p=2, dim=1)  # (test_image_num,512)
        embeds = embeds.to(torch.device("cpu"))
        distances = self.trainer_feature_density.predict(embeds)  # (image_num,)
        cur_dist = distances.detach().numpy()[0]
        # if distances.detach().numpy()[0] > self.dist_threshold_val:
        save_ng_dir = os.path.join(data_path, "ng")
        save_ok_dir = os.path.join(data_path, "bg")

        # distances_array = distances.numpy()
        # test_image_path = str(test_data_eval.image_names[i])
        print(test_file_path, cur_dist)
        if self.dist_threshold_val < cur_dist:
            save_full_path = test_file_path.replace(data_path, save_ng_dir)
            # Grad CAM
            # target_layers = [self.model.resnet18.layer4[-1]]
            target_layers = [self.model.resnet18.layer4]
            cam = GradCAM(model=self.model, target_layers=target_layers)
            # targets = [ClassifierOutputTarget(281)]     # cat
            # targets = [ClassifierOutputTarget(1)]  # dog
            targets = None  # dog

            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_resized = cv2.resize(image, (self.size, self.size))

            visualization = show_cam_on_image(img_resized.astype(dtype=np.float32) / 255.,
                                              grayscale_cam, use_rgb=True)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            output = np.hstack([visualization, img_resized])

            # plt.imshow(output)
            # plt.show()
        else:
            output = image
            save_full_path = test_file_path.replace(data_path, save_ok_dir)
        save_path_dir = os.path.dirname(save_full_path)
        os.makedirs(save_path_dir, exist_ok=True)

        img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_full_path, img_rgb)

    def preprocess_image(self, image):
        # 调整图像大小
        img_resized = cv2.resize(image, (self.size, self.size))
        # 将BGR转换为RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 归一化
        img_normalized = img_rgb / 255.0  # 先将像素值缩放到[0,1]之间
        mean = np.array(self.mean)
        std = np.array(self.std)
        img_normalized = (img_normalized - mean) / std

        # 转换为Tensor
        # transform_to_tensor = transforms.ToTensor()
        img_tensor = torch.tensor(img_normalized.astype(np.float32))
        img_tensor = torch.unsqueeze(img_tensor, 0).permute([0, 3, 1, 2])
        return img_tensor

    def _get_train_embeds(self):
        # train data / train kde
        test_data = MVTecAT(self.data_path, self.defect_type, self.size, transform=self.test_transform,
                            mode="train")

        dataloader_train = DataLoader(test_data, batch_size=64,
                                      shuffle=False, num_workers=0)
        train_embed = []
        with torch.no_grad():
            for x in dataloader_train:
                embed, logit = self.model(x.to(self.device))
                train_embed.append(embed.cpu())
        train_embed = torch.cat(train_embed)
        return train_embed

    # # 计算贴上后的patch周围的对比度
    # def calculate_contrast_after_paste(self, augmented, insert_box, margin=5):
    #     # 提取插入位置周围的区域
    #     w_start = max(insert_box[0] - margin, 0)
    #     h_start = max(insert_box[1] - margin, 0)
    #     w_end = min(insert_box[2] + margin, augmented.width)
    #     h_end = min(insert_box[3] + margin, augmented.height)
    #
    #     surrounding_area = augmented.crop((w_start, h_start, w_end, h_end))
    #
    #     # 使用ImageStat来获取周围区域的标准差，作为对比度的指标
    #     stats = ImageStat.Stat(surrounding_area)
    #     contrast = stats.stddev
    #
    #     return contrast


from pathlib import Path

if __name__ == '__main__':
        # all_types_th = {
        #     'bottle': 50,
        #     'cable': 50,
        #     'capsule': 100,
        #     'carpet': 100,
        #     'grid': 140,
        #     'hazelnut': 50,
        #     'leather': 200,
        #     'metal_nut': 100,
        #     'pill': 100,
        #     'screw': 30,
        #     'tile': 50,
        #     'toothbrush': 150,
        #     'transistor': 100,
        #     'wood': 120,
        #     'zipper': 100
        # }

        total_data_path = r"H:\zxq\data\zhongke_anomaly_detection"
        dataset_names = os.listdir(total_data_path)
        for dataset_name in dataset_names:

            # data_path = r"F:\zxq\data\mvtec_anomaly_detection"
            # data_path = r"F:\zxq\data\zk\zhongke_anomaly_detection\5AA"
            data_path = os.path.join(total_data_path, dataset_name)
            model_dir = os.path.basename(data_path)
            data_types = os.listdir(data_path)
            all_types_th = {}
            for data_type in data_types:
                all_types_th[data_type] = 30

            model_names = [list(Path(model_dir).glob(f"model-{data_type}*"))[0] for data_type in all_types_th.keys() if
                           len(list(Path(model_dir).glob(f"model-{data_type}*"))) > 0]
            if len(model_names) < len(all_types_th.keys()):
                print("warning: not all types present in folder")

            for model_name, data_type in zip(model_names, all_types_th):
                print(f"evaluating {data_type}")
                classifier = Classifier(data_path=data_path, model_path=model_name, defect_type=data_type, dist_threshold_val=all_types_th[data_type], size=256, head_layer=1)
                res_list = classifier.test_dataset()
                ok_num, ng_num, fp_num, fn_num, fp_rate, fn_rate = res_list
                header_list = ["ok num", "ng num", "fp num", "fn num", "fp rate", "fn rate"]
                data_list = [
                    {"ok num": ok_num, "ng num": ng_num, "fp num": fp_num, "fn num": fn_num, "fp rate": fp_rate, "fn rate": fn_rate}
                ]
                with open(dataset_name + "/{}.csv".format(data_type), mode="w", encoding="utf-8-sig", newline="") as f:
                    # 基于打开的文件，创建 csv.writer 实例
                    writer = csv.DictWriter(f, header_list)
                    # 写入 header
                    writer.writeheader()

                    # 写入数据
                    writer.writerows(data_list)

