# dataloader.py

from torch.utils.data import Dataset, DataLoader
import torch

# class TextDescriptionDataset(Dataset):
#     def __init__(self, data):
#         """
#         Args:
#             data (dict): A dictionary where keys are text descriptions and values are lists,
#                          with the first element being the positive description and the rest being negative samples.
#         """
#         self.samples = []
#         for text, descriptions in data.items():
#             pos_description = descriptions[0]
#             neg_descriptions = descriptions[1:]
#             for neg_description in neg_descriptions:
#                 if text and pos_description and neg_description:
#                     text = text.split(';;;')[-1]
#                     pos_description = pos_description.split(';;;')[-1]
#                     neg_description = neg_description.split(';;;')[-1]
#                     self.samples.append((text, pos_description, neg_description))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         text_description, description_pos, description_neg = self.samples[idx]
#         return {
#             "text_description": text_description,
#             "description_pos": description_pos,
#             "description_neg": description_neg
#         }
class TextDescriptionDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): A dictionary where keys are text descriptions and values are lists,
                         with the first element being the positive description and the rest being negative samples.
        """
        self.samples = []
        for text, descriptions in data.items():
            # 分割text_entity和text_description
            text_parts = text.split(';;;')
            text_entity = text_parts[0]
            text_description = text_parts[1] if len(text_parts) > 1 else ""

            pos_description = descriptions[0]
            neg_descriptions = descriptions[1:]

            for neg_description in neg_descriptions:
                if text_entity and pos_description and neg_description:
                    # 分割pos_entity和pos_description
                    pos_parts = pos_description.split(';;;')
                    pos_entity = pos_parts[0]
                    pos_description = pos_parts[1] if len(pos_parts) > 1 else ""

                    # 分割neg_entity和neg_description
                    neg_parts = neg_description.split(';;;')
                    neg_entity = neg_parts[0]
                    neg_description = neg_parts[1] if len(neg_parts) > 1 else ""
                    # print(111111111111111)
                    # print(text_entity)
                    # print(text_description)
                    # print(pos_entity)
                    # print(neg_entity)
                    # print(22222222222222)

                    self.samples.append({
                        "text_entity": text_entity,
                        "pos_entity": pos_entity,
                        "neg_entity": neg_entity,
                        "text_description": text_description,
                        "description_pos": pos_description,
                        "description_neg": neg_description
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_dataloader(data, batch_size=8, shuffle=True):
    dataset = TextDescriptionDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
