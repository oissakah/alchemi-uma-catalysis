from __future__ import annotations

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from nvalchemi.models.uma import UMAWrapper


def load_alchemi_uma(
    checkpoint: str = "uma-s-1p2p1",
    task: str = "oc20",
    device: str = "cuda",
    inference_settings: str = "default",
):
    return UMAWrapper.from_checkpoint(
        checkpoint,
        task_name=task,
        device=device,
        inference_settings=inference_settings,
    )


def load_fairchem_uma(
    checkpoint: str = "uma-s-1p2p1",
    task: str = "oc20",
    device: str = "cuda",
    inference_settings: str = "default",
):
    try:
        predictor = pretrained_mlip.get_predict_unit(
            checkpoint,
            device=device,
            inference_settings=inference_settings,
        )
    except TypeError:
        # Compatibility with fairchem versions where inference settings are
        # configured internally by get_predict_unit.
        predictor = pretrained_mlip.get_predict_unit(checkpoint, device=device)
    calc = FAIRChemCalculator(predictor, task_name=task)
    return predictor, calc
