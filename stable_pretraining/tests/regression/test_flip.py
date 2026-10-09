"""FLIP trains, validates, and resumes through the normal Manager lifecycle."""

from copy import deepcopy
import json
import sys

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.tests.unit.test_flip import (
    TinyTextEncoder,
    make_coco,
    tiny_flip,
)

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


def test_coco_example_trains_and_evaluates(tmp_path, monkeypatch):
    from examples import flip as example

    monkeypatch.setattr(spt.get_config(), "_progress_bar", "none")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    class Tokenizer:
        """Offline tokenizer for the example's complete training lifecycle."""

        def __len__(self):
            return 32

        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.tensor([[int(text) + 1] * 32 for text in texts])}

    monkeypatch.setattr(
        example.AutoTokenizer, "from_pretrained", lambda name: Tokenizer()
    )
    make_coco(tmp_path)
    evaluate = example.evaluate

    def check_final_split(model, loader, normalize):
        assert loader.dataset.image_ids == [6, 7, 8]
        assert all(path.parent.name == "val2017" for path in loader.dataset.paths)
        return evaluate(model, loader, normalize)

    monkeypatch.setattr(example, "evaluate", check_final_split)
    token_counts = []
    constructor = spt.FLIP

    def capture_model(*args, **kwargs):
        model = constructor(*args, **kwargs)

        def check_normalization(module, inputs):
            images = inputs[0]
            assert images.device.type == "cpu"
            assert images.dtype == torch.float32
            # The fixture has zero green/blue pixels, even after cropping.
            torch.testing.assert_close(
                images[:, 1], torch.full_like(images[:, 1], -0.456 / 0.224)
            )

        model.image_encoder.register_forward_pre_hook(check_normalization)
        model.image_encoder.vit.blocks[0].register_forward_pre_hook(
            lambda module, inputs: token_counts.append(
                (model.training, inputs[0].shape[1])
            )
        )
        return model

    monkeypatch.setattr(spt, "FLIP", capture_model)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flip.py",
            "--data",
            str(tmp_path),
            "--output",
            str(output),
            "--train-size",
            "4",
            "--dev-size",
            "2",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--epochs",
            "1",
        ],
    )
    example.main()
    assert (True, 99) in token_counts
    assert (False, 197) in token_counts
    assert all(count == (99 if training else 197) for training, count in token_counts)
    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics) == {"image_to_text", "text_to_image"}
    assert all(
        0 <= recall <= 1
        for direction in metrics.values()
        for recall in direction.values()
    )


def test_manager_training_validation_and_checkpoint(tmp_path):
    pl.seed_everything(11, workers=True)
    model = tiny_flip(optim={"optimizer": {"type": "AdamW", "lr": 1e-3}})
    samples = [
        {"image": torch.randn(3, 32, 32), "input_ids": torch.randint(0, 32, (6,))}
        for _ in range(8)
    ]
    data = spt.data.DataModule(
        train=DataLoader(samples, batch_size=4, drop_last=True),
        val=DataLoader(samples, batch_size=4),
    )
    initial = deepcopy(model.state_dict())
    trainer_args = dict(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
    )
    trainer = pl.Trainer(max_epochs=1, **trainer_args)
    spt.Manager(trainer=trainer, module=model, data=data, seed=11)()
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["fit/loss_epoch"])
    assert torch.isfinite(trainer.callback_metrics["validate/loss"])
    for key in ("image_projector.weight", "text_projector.weight", "logit_scale"):
        assert not torch.equal(model.state_dict()[key], initial[key])

    checkpoint = tmp_path / "flip.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = spt.FLIP.load_from_checkpoint(
        checkpoint,
        text_encoder=TinyTextEncoder(),
        text_dim=16,
        projection_dim=8,
        encoder_name=tiny_flip().image_encoder.vit,
        mask_ratio=0,
        optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
    )
    assert restored.hparams.mask_ratio == 0
    torch.testing.assert_close(restored.logit_scale, model.logit_scale)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        for key, value in model(batch, stage="validate").items():
            torch.testing.assert_close(restored(batch, stage="validate")[key], value)
    resumed = pl.Trainer(max_epochs=2, **trainer_args)
    restored.train()
    spt.Manager(
        trainer=resumed,
        module=restored,
        data=data,
        seed=11,
        ckpt_path=str(checkpoint),
        weights_only=False,
    )()
    assert resumed.global_step == 4
    for key in ("image_projector.weight", "image_encoder.vit.patch_embed.proj.weight"):
        assert not torch.equal(restored.state_dict()[key], model.state_dict()[key])
