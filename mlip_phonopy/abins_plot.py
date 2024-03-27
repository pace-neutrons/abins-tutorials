#! /usr/bin/env python3

from pathlib import Path
from typing import Optional

import mantid.simpleapi
from matplotlib import pyplot as plt
import numpy as np
import typer
from typing_extensions import Annotated

def main(
    filename: Path,
    output: Annotated[Path, typer.Argument()] = (Path.cwd() / "abins-plot.pdf"),
    ref_dat: Optional[Path] = None,
    ref_label: Optional[str] = None,
    ref_scale: float = 1.0,
    temperature: float = 25.,
) -> None:

    mantid.simpleapi.Abins(
        VibrationalOrPhononFile=str(filename),
        AbInitioProgram="FORCECONSTANTS",
        OutputWorkspace="abins-output",
        TemperatureInKelvin=temperature,
        SumContributions=True,
        ScaleByCrossSection="Total",
        QuantumOrderEventsNumber="2",
        Autoconvolution=True,
        Instrument="TOSCA",
        Setting="Backward (TOSCA)",
    )

    workspace = mantid.simpleapi.mtd["abins-output_total"]

    frequency_bins, intensities = workspace.extractX(), workspace.extractY()

    # Convert 2-D arrays with one row to 1-D arrays for plotting
    intensities = intensities.flatten()
    frequency_bins = frequency_bins.flatten()

    # Convert (N+1) bin edges to N midpoints to plot against N intensities
    frequency_midpoints = (frequency_bins[:-1] + frequency_bins[1:]) / 2

    fig, ax = plt.subplots()
    ax.plot(frequency_midpoints, intensities, label="TOSCA backscattering sim.")
    ax.set_xlabel(r"Frequency / cm$^{-1}$")
    ax.set_ylabel("Intensity")

    if ref_dat is None:
        ax.set_title("Simulated TOSCA spectrum (backscattering)")

    else:
        ref_workspace = mantid.simpleapi.Load(str(ref_dat))
        frequencies = ref_workspace.extractX()
        intensities = ref_workspace.extractY() * ref_scale

        if ref_label is None:
            ref_label = str(ref_dat)

        ax.plot(frequencies.flatten(), intensities.flatten(), label=ref_label)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output)


if __name__ == "__main__":
    typer.run(main)
