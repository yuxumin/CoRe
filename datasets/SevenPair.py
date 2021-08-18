import torch
import scipy.io
import os
import random
from utils import misc
from PIL import Image

class SevenPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform

        classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        self.sport_class = classes_name[args.class_idx - 1]

        self.class_idx = args.class_idx # sport class index(from 1 begin)
        self.score_range = args.score_range
        # file path
        self.data_root = args.data_root
        self.data_path = os.path.join(self.data_root, '{}-out'.format(self.sport_class))
        self.split_path = os.path.join(self.data_root, 'Split_4', 'split_4_train_list.mat')
        self.split = scipy.io.loadmat(self.split_path)['consolidated_train_list']
        self.split = self.split[self.split[:,0] == self.class_idx].tolist()
        if self.subset == 'test':
            self.split_path_test = os.path.join(self.data_root, 'Split_4', 'split_4_test_list.mat')
            self.split_test = scipy.io.loadmat(self.split_path_test)['consolidated_test_list']
            self.split_test = self.split_test[self.split_test[:,0] == self.class_idx].tolist()
        # setting
        self.length = args.frame_length
        self.voter_number = args.voter_number

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else:
            self.dataset = self.split.copy()

    def load_video(self, idx):
        video_path = os.path.join(self.data_path, '%03d'%idx)  
        video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % ( i + 1 ))) for i in range(self.length)]
        return self.transforms(video)

    def delta(self):
        delta = []
        dataset = self.split.copy()
        for i in range(len(dataset)):
            for j in range(i+1,len(dataset)):
                delta.append(
                    abs(
                    misc.normalize(dataset[i][2], self.class_idx, self.score_range) - 
                    misc.normalize(dataset[j][2], self.class_idx, self.score_range)))
        return delta

    def __getitem__(self,index):
        sample_1  = self.dataset[index]
        assert int(sample_1[0]) == self.class_idx
        idx = int(sample_1[1])

        data = {}
        if self.subset == 'test':
            # test phase
            data['video'] = self.load_video(idx)
            data['final_score'] = misc.normalize(sample_1[2], self.class_idx, self.score_range)
            # choose a list of sample in training_set
            train_file_list = self.split.copy()
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            #print(len(choosen_sample_list))
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp_idx = int(item[1])
                tmp['video'] = self.load_video(tmp_idx)
                tmp['final_score'] = misc.normalize(item[2], self.class_idx, self.score_range)
                target_list.append(tmp)
            return data , target_list
        else:
            # train phase
            data['video'] = self.load_video(idx)
            data['final_score'] = misc.normalize(sample_1[2], self.class_idx, self.score_range)
         
            # choose a sample
            # did not using a pytorch sampler, using diff_dict to pick a video sample

            file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            tmp_idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[tmp_idx]
            target = {}
            # sample 2
            target['video'] = self.load_video(int(sample_2[1]))
            target['final_score'] = misc.normalize(sample_2[2], self.class_idx, self.score_range)
            return data , target
    def __len__(self):
        return len(self.dataset)