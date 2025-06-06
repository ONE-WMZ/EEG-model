import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class Datasets_train(Dataset):
    def __init__(self, root_dir='../Data/test', window_size=400, overlap=350, transform=None):
        self.root_dir = os.path.normpath(root_dir)
        self.window_size = window_size
        self.step = window_size - overlap
        self.file_segments = []
        self.labels_map = {'label_0': 0, 'label_1': 1}
        self.transform = transform

        for label_name in self.labels_map:
            label_path = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            for filename in os.listdir(label_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(label_path, filename)
                    try:
                        data = np.load(file_path, mmap_mode='r')
                        data_length = data.shape[1]

                        if data_length < self.window_size:
                            print(f"Skipping {filename}: length={data_length} < window_size={self.window_size}")
                            continue

                        num_segments = max(0, (data_length - window_size) // self.step + 1)
                        for start in range(0, num_segments * self.step, self.step):
                            self.file_segments.append((file_path, start, self.labels_map[label_name]))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

    def __len__(self):
        return len(self.file_segments)

    def __getitem__(self, idx):
        while True:
            file_path, start, label = self.file_segments[idx]

            try:
                data = np.load(file_path, mmap_mode='r')
                if data.shape[1] <= start + self.window_size:
                    raise ValueError("Start index out of bounds")

                segment = data[:, start:start + self.window_size].copy()
                break
            except Exception as e:
                # print(f"Error loading index {idx}: {e}")
                idx = (idx + 1) % len(self)

        if self.transform:
            segment = self.transform(segment)

        segment = torch.from_numpy(segment).float()
        label = torch.tensor(label).long()

        return segment, label

# ------------------------------------------------------------------------
# 直接调用
test_dataset = Datasets_train()
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
# ------------------------------------------------------------------------

# 测试用例
# if __name__ == "__main__":
#     dataset = Datasets_train()
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
#
#     for i, (segments, labels) in enumerate(dataloader):
#         print(i)
#         # print(f"Batch {i}: Segments shape={segments.shape}, Labels={labels}")
