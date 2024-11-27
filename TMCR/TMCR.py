"""
@DESCRIPTION: TMCR testing on AVA
@AUTHOR: yzc
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import warnings
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.models.swin_transformer import swin_b
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights
from sklearn.metrics import mean_squared_error
use_gpu = True
warnings.filterwarnings("ignore")


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


class GIAA_model(nn.Module):
    def __init__(self, Aesthetic_Level_extractor):
        super(GIAA_model, self).__init__()

        self.AL = Aesthetic_Level_extractor

        self.TF_Fea_shift = nn.Sequential(nn.Linear(3804, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))
        self.AL_Fea_shift = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))

        self.regression = nn.Sequential(nn.Linear(2048, 10), nn.Softmax(dim=1))

    def forward(self, x, TM_feature):
        TF_feature = self.TF_Fea_shift(TM_feature)

        AL_feature = self.AL(x)
        AL_feature = self.AL_Fea_shift(AL_feature)

        TFAL_feature = torch.cat((TF_feature, AL_feature), dim=1)

        score = self.regression(TFAL_feature)

        return score


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class ImageAVADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_information = pd.read_csv(csv_file, sep=',')[1:]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_information)

    def __getitem__(self, idx):
        img_path = str(os.path.join(self.root_dir, (str(self.image_information.iloc[idx, 0]) + '.jpg')))
        image = Image.open(img_path).convert('RGB')
        label = self.image_information.iloc[idx, 1:11]
        TM_feature = self.image_information.iloc[idx, 11]
        TM_feature = TM_feature[1:-1]
        TM_feature = [float(x) for x in TM_feature.split(',')]
        y = np.array([label[k] for k in range(10)])
        p = y / y.sum()
        if self.transform:
            image = self.transform(image)
        sample = {'image': image,
                  'label': torch.from_numpy(np.float64([p])).double(),
                  'TM_feature': torch.from_numpy(np.float64([TM_feature])).double()}
        return sample


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data():
    data_image_dir = os.path.join(r'./AVA/images')
    data_image_test_label_dir = os.path.join(r'./TMCR/test_TM.csv')

    transformed_dataset_test = ImageAVADataset(
        csv_file=data_image_test_label_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),
            ]
        )
    )
    data_test = DataLoader(transformed_dataset_test, batch_size=128,
                           shuffle=False, num_workers=3, collate_fn=my_collate, drop_last=False)

    return data_test


def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def test():
    # data
    data_valid = load_data()

    CR_model = Rank_Swin()
    model = GIAA_model(CR_model.encoder_image)
    checkpoint = torch.load(r'./Model/TMCR_AVA.pt')
    model.load_state_dict(checkpoint)

    model.cuda()

    print('***********************valid***********************')
    model.eval()
    predicts_score = []
    ratings_score = []
    for batch_idx, data in enumerate(data_valid):
        inputs = data['image']
        batch_size = inputs.size()[0]
        labels = data['label'].view(batch_size, -1)
        TM_features = data['TM_feature'].view(batch_size, -1)

        inputs, labels, TM_features = Variable(inputs.float().cuda()), Variable(labels.float().cuda()), Variable(TM_features.float().cuda())

        with torch.no_grad():
            outputs = model(inputs, TM_features)

        predtions, predtions_np = get_score(outputs)
        ratings, ratings_np = get_score(labels)

        predicts_score += predtions_np.tolist()
        ratings_score += ratings_np.tolist()

    plcc = pearsonr(predicts_score, ratings_score)[0]
    srcc = spearmanr(predicts_score, ratings_score)[0]
    mse = mean_squared_error(predicts_score, ratings_score)

    print('Valid Regression SRCC:%4f' % srcc)
    print('Valid Regression PLCC:%4f' % plcc)
    print('MSE:%4f' % mse)


if __name__ == '__main__':
    test()