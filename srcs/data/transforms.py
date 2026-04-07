from monai.transforms import Compose, NormalizeIntensityd, ResizeWithPadOrCropd


def build_hcp_transforms(target_spatial_size, target_depth=160):
    return Compose([
        ResizeWithPadOrCropd(
            keys=["t1", "t2"],
            spatial_size=(target_depth, *target_spatial_size),
        ),
        NormalizeIntensityd(
            keys=["t1", "t2"],
            nonzero=True,
            channel_wise=True,
        ),
    ])
