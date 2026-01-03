import torch

from src.model_architecture import vulcanicModel

checkpoint_path = (
    "./vulcanology/s3chvss7/checkpoints/epoch=epoch=8-val_loss=val_loss=1.8303.ckpt"
)
model = vulcanicModel.load_from_checkpoint(checkpoint_path, features=89)

dummy = torch.randn(1, 89)
torch.onnx.export(model, dummy, "model.onnx")
