from os import path
from glob import glob
import numpy as np
import skvideo.io
import pandas as pd
from tqdm import tqdm
import csv

import torch
from torch.utils import data
from torchvision import transforms


class VideoDataset(data.Dataset):
    def __init__(self,
                 root,
                 mode='val',
                 transform=None,
                 seq_len=32,
                 num_seq=2,
                 downsample=1,
                 checklength=True,
                 ):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.toPIL = transforms.ToPILImage()

        if self.mode == 'train':
            # train_path = path.join(root, 'train')
            train_path = root
            if not checklength:
                self.train_split = glob(path.join(train_path, '*.*'))
            else:
                v_info_file = path.join(root, 'v_info.csv')
                if not path.isfile(v_info_file):
                    print('create video info file ...')
                    video_paths = glob(path.join(train_path, '*.*'))
                    with open(v_info_file, 'w') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        for video_path in tqdm(video_paths):
                            metadata = skvideo.io.ffprobe(video_path)
                            length = int(metadata['video']['@nb_frames'])
                            csv_writer.writerow([path.basename(video_path), length])
                video_info = pd.read_csv(v_info_file, header=None)
                self.train_split = []
                print('filter out too short videos ...')
                for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
                    vname, vlen = row[0], row[1]
                    if vlen - 2 * self.seq_len * self.downsample > 0:
                        self.train_split.append(path.join(train_path, vname))
            print('total number of videos: {}'.format(len(self.train_split)))
        elif self.mode == 'val':
            val_path = path.join(root, 'val')
            self.val_split = glob(path.join(val_path, '*.*'))
        elif self.mode == 'test':
            test_path = path.join(root, 'test')
            self.test_split = glob(path.join(test_path, '*.*'))
        else:
            raise ValueError('wrong mode')

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
        """
        if self.mode == 'train':
            video_name = self.train_split[idx]
        elif self.mode == 'val':
            video_name = self.val_split[idx]
        else:
            video_name = self.test_split[idx]

        video_data = skvideo.io.vread(video_name)
        length, width, height, channel = video_data.shape
        frames_indices = self.idx_sampler(length, video_name)
        clip = video_data[frames_indices]

        if self.transform:
            seq_clip = [self.toPIL(frame) for frame in clip]
            trans_clip = self.transform(seq_clip)

            # fix seed, apply the sample `random transformation` for all frames in the clip
            # seed = random.random()
            # for frame in clip:
            #     random.seed(seed)
            #     frame = self.toPIL(frame)  # PIL image
            #     frame = self.transform(frame)  # [C x H x W]
            #     trans_clip.append(frame)

            # # Seperate middle frames
            # mid_frame1 = trans_clip[int(self.seq_len/2)]
            # mid_farme2 = trans_clip[len(trans_clip) - int(self.seq_len/2)]

            (C, H, W) = trans_clip[0].size()
            clip = torch.stack(trans_clip)  # (T x C X H x W)
            clip = clip.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)  # num_seq, C, seq_length, H, W
        else:
            clip = torch.tensor(clip)

        # Seperate middle frames
        mid_idx = int(self.seq_len/2)
        # mid_idx2 = self.num_seq*self.seq_len - mid_idx1
        if self.num_seq == 2:
            frame1 = clip[0, :, mid_idx, :, :]
            frame2 = clip[1, :, mid_idx, :, :]
            frames = torch.stack([frame1, frame2])
        else:
            frames = clip[0, :, mid_idx, :, :].unsqueeze(0)

        return clip, frames, idx

    def idx_sampler(self, vlen, vname):
        """sample indices from a video"""
        if vlen-self.num_seq*self.seq_len*self.downsample > 0:
            start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
            seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        else:
            print('Video is too short: {}'.format(vname))

        return seq_idx

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_split)
        elif self.mode == 'val':
            return len(self.val_split)
        else:
            return len(self.test_split)


