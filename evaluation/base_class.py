import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

class GeneratedDataset(Dataset):
    def __init__(self,
                 eval_folder: str,
                 ):
        self.eval_folder = eval_folder

        self.prompts = []  # [L]
        self.src_img_paths = []  # [N*[2]]
        self.gen_img_folders = []  # [L]
        self.init()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        src_img_path = self.src_img_paths[index]
        gen_img_folder = self.gen_img_folders[index]

        src_img_tensor = self._img_path_to_nchw(src_img_path, size=512)  # (1,C,H,W)
        gen_img_tensors = [self._img_path_to_nchw(
            os.path.join(gen_img_folder, x)
        ) for x in os.listdir(gen_img_folder) if not x.endswith('txt')]  # (RB,C,H,W), R=repeats, B=batch_size
        gen_img_tensors = torch.stack(gen_img_tensors, dim=0)

        return {
            'prompt': prompt,
            'src_img_tensor': src_img_tensor,
            'gen_img_tensors': gen_img_tensors
        }

    def init(self):
        for dir in os.listdir(self.eval_folder): 
            root = os.path.join(self.eval_folder, dir)
            if not os.path.isdir(root):
                continue
            src_path = os.path.join(root, f"{dir}.jpg")
            for run in os.listdir(root):
                sub_root = os.path.join(root, run)
                if run == "logs" or not os.path.isdir(sub_root):
                    continue
                # gen_path = os.path.join(sub_root, "generated_img.png")
                with open(os.path.join(sub_root, "prompt_without_ph.txt"), 'r') as f:
                    prompt = f.readline().strip()
                self.src_img_paths.append(src_path)
                self.gen_img_folders.append(sub_root)
                self.prompts.append(prompt)

        print('[GeneratedDataset] txt files loaded.')

    def _img_path_to_nchw(self, img_path: str, size=None):
        image = Image.open(img_path).convert('RGB')
        if size is not None:
            image = image.resize((size, size))
        image = self.trans(image)
        return image

    @staticmethod
    def _img_paths_to_nchw(img_paths: str):
        images = [(np.array(Image.open(path).convert('RGB')) / 127.5 - 1.0).astype(np.float32) for path in img_paths]
        images = [torch.from_numpy(x).permute(2, 0, 1) for x in images]
        images = torch.stack(images, dim=0)  # (N,C,H,W)
        return images


class EvaluatorBase(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, gen: torch.Tensor, src: torch.Tensor, txt: str):
        pass


class IDCLIPScoreCalculator(ABC):
    def __init__(self,
                 eval_folder: str,
                 id_clip_evaluator: EvaluatorBase,
                 device: torch.device = torch.device('cuda:0'),
                 ):
        self.eval_folder = eval_folder
        self.device = device

        self.gen_dataset = None
        self.data_loader = None
        self._get_dataset_dataloader()

        self.id_clip_evaluator = id_clip_evaluator

    def _get_dataset_dataloader(self):
        gen_dataset = GeneratedDataset(
            self.eval_folder,
        )
        data_loader = DataLoader(
            gen_dataset, shuffle=False, batch_size=1, num_workers=4, drop_last=False
        )  # batch_size should be 1
        self.gen_dataset = gen_dataset
        self.data_loader = data_loader

    @torch.no_grad()
    def start_calc(self):
        sim_img_list = []
        sim_text_list = []
        id_cos_sim_list = []
        id_mse_dist_list = []
        id_l2_dist_list = []
        num_has_face, num_no_face = 0, 0
        for data in tqdm(self.data_loader):
            src = data['src_img_tensor'].cuda()
            gen = data['gen_img_tensors'].squeeze(dim=0).cuda()
            prompt = data['prompt']

            sim_img, sim_text, id_result_dict = self.id_clip_evaluator.evaluate(
                gen.repeat(1, 1, 1, 1),
                src.repeat(1, 1, 1, 1),
                prompt[0].replace('<sks>_v0 <sks>_v1', ''))

            id_cos_sim = id_result_dict["cos_sim"]
            id_mse_dist = id_result_dict["mse_dist"]
            id_l2_dist = id_result_dict["l2_dist"]
            has_face = id_result_dict["num_has_face"]
            no_face = id_result_dict["num_no_face"]

            # print("Image similarity: ", sim_img)
            # print("Text similarity: ", sim_text)
            # print("Identity cos similarity: ", id_cos_sim)
            sim_img_list.append(sim_img)
            sim_text_list.append(sim_text)
            if id_cos_sim > 1e-6:
                id_cos_sim_list.append(id_cos_sim)
                id_mse_dist_list.append(torch.FloatTensor([id_mse_dist]))
                id_l2_dist_list.append(torch.FloatTensor([id_l2_dist]))
            num_has_face += has_face
            num_no_face += no_face

        sim_img_avg = torch.stack(sim_img_list, dim=0)
        sim_text_avg = torch.stack(sim_text_list, dim=0)
        id_cos_sim_avg = torch.stack(id_cos_sim_list, dim=0)
        id_mse_dist_avg = torch.stack(id_mse_dist_list, dim=0)
        id_l2_dist_avg = torch.stack(id_l2_dist_list, dim=0)
        print("Image similarity (avg): ", sim_img_avg.mean())
        print("Text similarity (avg): ", sim_text_avg.mean())
        print("Identity cos similarity (avg): ", id_cos_sim_avg.mean(),
              f"mse_dist={id_mse_dist_avg.mean():.4f}, l2_dist={id_l2_dist_avg.mean():.4f}",
              f"has_face={num_has_face}, no_face={num_no_face}")
