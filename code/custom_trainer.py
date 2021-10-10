from sadice import SelfAdjDiceLoss
from transformers import Trainer


class TrainerWithDiceLoss(Trainer):
    """
        compute_loss를 DiceLoss 함수를 응용하기 위한 Custom Trainer
        DiceLoss를 통해 Data Imbalance 해결
        https://github.com/fursovia/self-adj-dice
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        criterion = SelfAdjDiceLoss()
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
