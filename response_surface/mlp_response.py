import torch
import numpy as np
from torch import nn
from torch.functional import Tensor
from sklearn.metrics import classification_report
from pytorch_lightning import LightningModule


class MLPResponse(LightningModule):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 100,
        strain = np.arange(100),
        dropout: float = 0.1,
    ):
        super().__init__()
        cuda0 = torch.device('cuda:0')
        self.strain = torch.tensor(strain, requires_grad=False, dtype=torch.float, device=cuda0)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, int(output_dim*0.3)),
            nn.ReLU(),
            nn.Linear(int(output_dim*0.3), int(output_dim*0.6)),
            nn.ReLU(),
            nn.Linear(int(output_dim*0.6), int(output_dim*0.8)),
            nn.ReLU(),
            nn.Linear(int(output_dim*0.8), output_dim),
        )

        # Criterion
        self.criterion = nn.MSELoss()
        
    def forward(self,x):
        x1 = self.decoder(x)
        #x2 = torch.mul(x1, self.strain)
        return x1

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=1e-3, weight_decay=1e-3)

    def training_step(self, batch: list, batch_idx: int):
        params, stress = batch
        stress_preds = self(params)
        loss = self.criterion(stress_preds, stress)
        self.log("Loss/train", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: list, batch_idx: int):
        params, stress = batch
        with torch.no_grad():
            stress_preds = self(params)
            loss = self.criterion(stress_preds, stress)
        self.log("Loss/validation", loss)
        return params, loss

    def validation_epoch_end(self, batch_parts):
        for params, loss in batch_parts:
            print(f"Loss for parameters: {params} = {loss}")