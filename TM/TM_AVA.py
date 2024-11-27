"""
@DESCRIPTION: AVA generate TM_feature
@AUTHOR: yzc
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:2048'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import clip
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate


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
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        return sample


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(batch_size=32):
    data_train_label_dir = r"./AVA/Label/AVA_Regular_Label/test.csv"
    data_image_dir = r"./AVA/images"

    transformed_dataset_train = ImageAVADataset(
        csv_file=data_train_label_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )

    data_train = DataLoader(transformed_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=my_collate, drop_last=False)

    return data_train


def save_feature(feature_list, feature_path):
    feature_dic = {

    }
    for i, feature in enumerate(feature_list):
        feature_dic[str(i)] = feature

    torch.save(feature_dic, feature_path)
    pass


def load_feature(feature_path):
    feature = torch.load(feature_path)
    feature_list = []
    for i in range(len(feature.keys())):
        feature_list.append(feature[str(i)])

    return feature_list


def generate_tags():
    semantic_csv = pd.read_csv(
        r'./TM/Sem_Tags.csv')
    aesthetic_csv = pd.read_csv(
        r'./TM/Attr_Tags.csv')
    semantic_tags = np.array(semantic_csv)
    aesthetic_tags = np.array(aesthetic_csv)
    all_tags = np.concatenate((semantic_tags, aesthetic_tags))
    all_tags = np.squeeze(all_tags)
    all_tags = np.concatenate((all_tags, all_tags))
    print(all_tags.shape)

    return all_tags


def generate_similarity():
    Text_Feature_extractor, preprocess = clip.load('ViT-L/14@336px')
    Text_Feature_extractor.float()
    Text_Feature_extractor.cuda()
    model = Text_Feature_extractor
    tags = clip.tokenize(generate_tags()).cuda()
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    data_train = load_data()
    similarity_feature = []
    loop = tqdm(enumerate(data_train), total=len(data_train), leave=True)
    for batch_idx, data in loop:
        inputs = data['image']
        inputs = Variable(inputs.float().cuda())

        with torch.no_grad():
            outputs, _ = model(inputs, tags)
        print(outputs.size())
        outputs = outputs.data.cpu().numpy()
        outputs = np.squeeze(outputs)
        similarity_feature.append(outputs)
    all_features = np.vstack(similarity_feature)

    return(pd.DataFrame(all_features))


def concat(TM_feature):
    all_data = pd.read_csv(r"./AVA/Label/AVA_Regular_Label/test.csv")
    TM_features_filter = TM_features.apply(lambda row: [float(x) for x in row], axis=1)
    all_data['TM_features'] = TM_features_filter
    col1 = 'tag1'  
    col2 = 'TM_features'  
    columns = list(all_data.columns)

    index1 = columns.index(col1)
    index2 = columns.index(col2)

    columns[index1], columns[index2] = col2, col1

    all_data = all_data[columns]

    all_data.to_csv('./TMCR/test_TM.csv', index=False)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    TM_feature = generate_similarity()
    concat(TM_feature)
