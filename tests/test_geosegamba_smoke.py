import torch

from models import GeoSegambaConfig, build_geosegamba


def main() -> None:
    model = build_geosegamba(
        in_channels=4,
        num_classes=7,
        geo_prior_channels=1,
        dims=(32, 64, 128, 160),
        depths=(1, 1, 1, 1),
        decoder_channels=64,
    )
    model.eval()

    image = torch.randn(2, 4, 256, 256)
    geo_prior = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        output = model(image, geo_prior=geo_prior, return_details=True)

    assert output["logits"].shape == (2, 7, 256, 256)
    assert output["f1"].shape[-2:] == (128, 128)
    assert output["f4"].shape[-2:] == (16, 16)
    assert output["geoss"]["path_weights"].shape == (2, 3)

    config = GeoSegambaConfig(in_channels=3, num_classes=6)
    model = build_geosegamba(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        geo_prior_channels=config.geo_prior_channels,
        dims=config.dims,
        depths=config.depths,
        decoder_channels=config.decoder_channels,
    )
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 128, 128), geo_prior=torch.randn(1, 1, 128, 128))

    assert logits.shape == (1, 6, 128, 128)
    print("GeoSegamba smoke test passed.")


if __name__ == "__main__":
    main()
