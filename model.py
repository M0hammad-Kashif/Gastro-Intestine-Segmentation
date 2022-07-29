import segmentation_models_pytorch as smp

from config import CFG


def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=CFG.backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=CFG.num_classes,
        activation=None,
    )
    model.to(CFG.device)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
