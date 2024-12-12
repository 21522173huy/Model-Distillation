
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, test_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader
        self._test_dataloader = test_dataloader

    def get_train_dataloader(self):
        if self._train_dataloader is not None:
            return self._train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_dataloader is not None:
            return self._eval_dataloader
        return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset=None):
        if self._test_dataloader is not None:
            return self._test_dataloader
        return super().get_test_dataloader(test_dataset)
