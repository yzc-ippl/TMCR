"""
@DESCRIPTION: Contrastive Ranking on AVA
@AUTHOR: yzc
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
import pandas as pd
import numpy as np
import warnings
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
from torch.autograd import Variable
from torch import nn
from PIL import Image
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights
from torch.utils.data.dataloader import default_collate
import clip
warnings.filterwarnings("ignore")


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class Rank_Swin(nn.Module):
    def __init__(self):
        super(Rank_Swin, self).__init__()
        self.encoder_image = swin_b()
        self.encoder_image.load_state_dict(torch.load(r'./Model/swin_b-68c6b09e.pth'))
        self.encoder_image.head = nn.Sequential()

        self.projector_image = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
                                             nn.Linear(1024, 1024))

    def forward(self, image):
        image_feature = self.encoder_image(image)
        image_feature_contrast = self.projector_image(image_feature)
        feature = nn.functional.normalize(image_feature_contrast, dim=1)

        return feature


class ImageAVADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_information = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_information)

    def __getitem__(self, idx):
        img_path = str(os.path.join(self.root_dir, (str(self.image_information.iloc[idx, 0]) + '.jpg')))
        image = Image.open(img_path).convert('RGB')
        label = self.image_information.iloc[idx, 1:11]
        semantic_label = self.image_information.iloc[idx, 14]
        y = np.array([k for k in label])
        p = y / y.sum()
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': torch.from_numpy(np.float64(p)).double(),'semantic_label': semantic_label}
        return sample


def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def generate_semantic_score_dic():
    all_data = pd.read_csv("./train_semantic.csv", sep=',')
    semantic_label = np.array(all_data['Semantic'])
    aesthetic_score = np.array(all_data['Aesthetic'])
    score_level = [2, 3, 4, 5, 6, 7, 8, 9]
    all_data_dic = {}

    for semantic in range(15):
        all_data_dic[str(semantic)] = {}
        for i in range(1, len(score_level)):
            all_data_dic[str(semantic)][score_level[i - 1]] = []

    for j in range(15):
        semantic_idx = np.where(semantic_label == j)[0]
        semantic_idx_score = aesthetic_score[semantic_idx]
        for i in range(1, len(score_level)):
            semantic_idx_score_idx = \
                np.where((semantic_idx_score > score_level[i - 1]) & (semantic_idx_score < score_level[i]))[0]
            if len(semantic_idx_score_idx) == 0:
                continue
            else:
                all_data_dic[str(j)][score_level[i - 1]] = semantic_idx[semantic_idx_score_idx]

    return all_data_dic


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_samplers, sampling_num):
        """
        :param labels: semantic_score_dic
        :param n_samplers:  bin_size
        :param sampling_num: number of sampling for each semantic
        """

        self.labels = labels
        self.n_samplers = n_samplers
        self.sampling_num = sampling_num

    def __iter__(self):
        count = 0
        while count < self.sampling_num:
            indices = []
            for key in self.labels.keys():
                if len(self.labels[key]) == 0:
                    continue
                if len(self.labels[key]) < self.n_samplers:
                    sample = np.random.choice(self.labels[key], self.n_samplers, replace=True)
                if len(self.labels[key]) >= self.n_samplers:
                    sample = np.random.choice(self.labels[key], self.n_samplers, replace=False)
                indices.extend(sample)
            yield indices
            count += 1

    def __len__(self):
        return self.sampling_num


def ContrastBatch(semantic='0', bin_size=9, sampling_num=30):
    all_data_dic = generate_semantic_score_dic()
    # print(all_data_dic[semantic])
    semantic_label = all_data_dic[semantic]
    train_batch_sampler = BalancedBatchSampler(semantic_label, bin_size, sampling_num)
    data_image_dir = os.path.join(r'./AVA/images')
    data_image_train_label_dir = r"./train_semantic.csv"
    transformed_dataset_train = ImageAVADataset(
        csv_file=data_image_train_label_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )

    data_train = DataLoader(transformed_dataset_train, batch_sampler=train_batch_sampler, num_workers=3,
                            collate_fn=my_collate)

    return data_train


# aesthetic contrast
def make_criterion(batch_size, temperature):
    """
    :param batch_size:
    :param temperature:
    :return:
    """

    def criterion(features, labels):
        """
        :param features:  [batch_size, view, feature_dim]
        :param label: [batch_size, label]
        :return:
        """

        # logits
        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = anchor_feature
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask
        mask = torch.eq(labels, labels.T)
        contrast_count = features.shape[1]
        anchor_count = contrast_count
        mask = mask.repeat(anchor_count, contrast_count)
        # print(anchor_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    return criterion


def aesthetic_contrast(features, bin_num=9):
    """
    :param features: [batch_size, features]  semantic_num * bin_size
    :param bin_num: bin_size default 20
    :return:
    """

    contrast_feature = []
    contrast_label = []
    num_bin = int(features.shape[0] / bin_num)
    bin_size = [9 for i in range(num_bin)]

    # from left to right
    for anchor in range(num_bin - 1):
        for pos in range(anchor, num_bin - 1):
            old_label = [bin_size[i] for i in range(anchor)]
            pos_label = [1] * (bin_size[anchor] + bin_size[pos])
            neg_bins = [bin_size[i] for i in range(pos + 1, num_bin)]
            neg_label = [2] * sum(neg_bins)

            pos_features = torch.cat([features[sum(old_label): sum(old_label) + bin_size[anchor]],
                                      features[
                                      sum(bin_size) - sum(neg_bins) - bin_size[pos]: sum(bin_size) - sum(neg_bins)]],
                                     dim=0)
            neg_features = features[sum(bin_size) - sum(neg_bins):]
            anchor_contrast_label = pos_label + neg_label
            anchor_contrast_feature = torch.cat([pos_features, neg_features], dim=0)
            contrast_feature.append(anchor_contrast_feature)
            contrast_label.append(anchor_contrast_label)

    # from right to left
    for anchor in range(num_bin-1, 0, -1):
        for pos in range(anchor, 0, -1):
            old_label = [bin_size[i] for i in range(num_bin-1, anchor, -1)]
            pos_label = [1] * (bin_size[anchor] + bin_size[pos])
            neg_bins = [bin_size[i] for i in range(pos-1, -1, -1)]
            neg_label = [2] * sum(neg_bins)
            pos_features = torch.cat(
                [
                    features[sum(bin_size)-sum(old_label)-bin_size[anchor]: sum(bin_size)-sum(old_label)],
                    features[sum(neg_bins): sum(neg_bins)+bin_size[pos]]
                ],
                dim=0
            )
            neg_features = features[0: sum(neg_bins)]
            anchor_contrast_label = pos_label + neg_label
            anchor_contrast_feature = torch.cat([pos_features, neg_features], dim=0)
            contrast_feature.append(anchor_contrast_feature)
            contrast_label.append(anchor_contrast_label)

    supCR_loss = 0.0
    for feature, label in zip(contrast_feature, contrast_label):
        label = torch.from_numpy(np.array(label)).double()
        label = Variable(label.cuda())
        batch_size, _ = feature.shape
        # print(batch_size)
        # print(batch_size)
        # print(label)
        feature = torch.unsqueeze(feature, 1)
        label = torch.unsqueeze(label, 1)
        criterion = make_criterion(batch_size, 0.07)
        supCR_loss += criterion(feature, label)

    return supCR_loss


def train_all():
    # parameter
    aesthetic_contrast_epoch_num = 50
    best_loss = float('inf')

    # model
    model = Rank_Swin()
    model.cuda()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=5E-2)

    # save_fig = 1

    # aesthetic contrast
    for epoch in range(aesthetic_contrast_epoch_num):
        epoch_loss = 0.0

        print('*****************contrast train*****************')
        model.train()
        for semantic in range(15):
            epoch_semantic_loss = 0.0
            data_contrast = ContrastBatch(semantic=str(semantic))

            for batch_idx, data in enumerate(data_contrast):
                inputs = data['image']
                inputs = Variable(inputs.float().cuda())

                features = model(inputs)
                # print(features.size())

                loss = aesthetic_contrast(features)
                print('aesthetic_contrast_loss: ', loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_semantic_loss += loss

            epoch_loss += epoch_semantic_loss

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print('**************************************************')
            print(best_loss)
            print('**************************************************')
            best_model = copy.deepcopy(model)
            torch.save(best_model, './model/RC-Swin.pt')


if __name__ == '__main__':
    train_all()
