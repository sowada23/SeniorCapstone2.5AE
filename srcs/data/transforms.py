from monai.transforms import Compose, NormalizeIntensityd, Resized


def build_hcp_transforms(target_spatial_size, target_depth=160):
    return Compose([
        # Resize the full volume to the model input size instead of center-cropping.
        # ResizeWithPadOrCropd trims anatomy when the source scan is larger than the
        # target shape, which is why superior/inferior brain regions can disappear
        # from exported slices.
        Resized(
            keys=["t1", "t2"],
            spatial_size=(target_depth, *target_spatial_size),
            mode=("trilinear", "trilinear"),
            anti_aliasing=True,
        ),
        NormalizeIntensityd(
            keys=["t1", "t2"],
            nonzero=True,
            channel_wise=True,
        ),
    ])
