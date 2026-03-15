"""
Embodied fly: integrate the PyTorch LIF whole-brain model with a
NeuroMechFly v2 (flygym) physical body on a 15 ms synchronisation cycle.

Architecture (as per the Eon Systems blog post):
  1. Sensory events from the virtual world -> identified sensory neuron rates.
  2. Brain activity is updated in the connectome-constrained LIF model.
  3. Selected descending neuron (DN) outputs -> low-dimensional motor commands.
  4. Body steps forward -> sensory state changes -> loop.

The sync ticker is non-real-time: we fully resolve each brain sub-step and
each physics sub-step before advancing the clock.  Each tick is 15 ms of
simulated time, which equals 150 LIF steps at dt=0.1 ms.

Memory notes
------------
We run batch_size=1 and discard neuron spike history immediately after
extracting DN spike counts.  The sparse weight matrix (~290 MB) is the
dominant allocation.  Total RAM usage stays well under 8 GB on CPU.

Usage
-----
    python code/embodied_fly.py --duration 5.0 --render
    python code/embodied_fly.py --duration 60.0 --no-render
"""

import sys
import os
import argparse

# ── Dual-libomp guard ────────────────────────────────────────────────────────
# MuJoCo ships its own libomp.dylib; PyTorch / numba ship another.  When numba
# JIT-compiles a function it spawns OpenMP worker threads via omppool which
# uses *its* libomp, but MuJoCo has already initialised *its* libomp through
# the vision renderer.  The two runtimes share no state, so the newly spawned
# worker thread derferences a null pointer and the process segfaults (ARM64
# SIGSEGV at 0x580).  Disabling numba JIT at process start prevents omppool
# from ever loading, sidestepping the conflict entirely.  The benchmark module
# (and therefore the brain model) still works; numba just falls back to pure
# Python which is fast enough for the offline connectivity lookups.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from time import time

import numpy as np

from flygym import Fly, Camera
from flygym.arena import FlatTerrain
from flygym.simulation import Simulation

import torch

# Make the project code directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pytorch import (
    MODEL_PARAMS,
    DT as _RUN_DT,
    TorchModel,
    get_hash_tables,
    get_weights,
)
from benchmark import path_comp, path_con, path_wt


# ============================================================================
# Constants
# ============================================================================

# Simulated time per synchronisation tick (milliseconds).
TICK_MS: float = 15.0

# LIF model timestep (ms) — must match run_pytorch.DT.
DT: float = 0.1

# Number of LIF timesteps per tick  (TICK_MS / DT).
STEPS_PER_TICK: int = int(TICK_MS / DT)  # 150 steps at DT=0.1 ms

# ============================================================================
# Descending Neuron (DN) FlyWire IDs (from published literature)
#
# IDs sourced from example.ipynb (left/right pairs where applicable):
#   DNa01, DNa02  -> yaw / turning (Yang et al., 2024)
#   oDN1          -> forward velocity (Sapkal et al., 2024)
#   P9            -> forward walking command (upstream of oDN1)
#   MN9           -> proboscis extension / feeding motor neuron (Shiu et al.)
#   aDN1          -> antennal grooming (Hampel et al., 2020)
#   MDN           -> backward walking / moonwalker escape
# ============================================================================

DESCENDING_NEURONS: dict[str, list[int]] = {
    # Steering / turning — left and right copies
    "dna01": [720575940627787609, 720575940644438551],
    "dna02": [720575940629327659, 720575940604737708],
    # Forward velocity command (P9_oDN1 left/right)
    "odn1": [720575940626730883, 720575940620300308],
    # P9 neurons — primary forward walking command (drives oDN1)
    "p9": [720575940627652358, 720575940635872101],
    # Proboscis extension / feeding motor neuron MN9 — left/right
    "mn9": [720575940660219265, 720575940618238523],
    # Antennal descending neuron (grooming) — left/right
    "adn1": [720575940616185531, 720575940624319124],
    # Moonwalker descending neurons — backward walking / escape
    "mdn": [
        720575940616026939,
        720575940631082808,
        720575940640331472,
        720575940610236514,
    ],
    # Giant Fibre — fast escape / startle response
    "giant_fibre": [720575940622838154, 720575940632499757],
}

# ============================================================================
# Sensory neuron FlyWire IDs
#
# Sugar GRN list matches the 23-neuron set used in example.ipynb.
# P9 input neurons (the primary upstream forward-walking command neurons)
# are registered as a second sensory group so they can be activated by
# taste cues, matching the blog post description of taste-driven locomotion.
# ============================================================================

SENSORY_NEURONS: dict[str, list[int]] = {
    # Labellar sugar gustatory receptor neurons — 200 Hz activation in notebook.
    "sugar_grn": [
        720575940616885538, 720575940630233916, 720575940639332736,
        720575940632889389, 720575940617000768, 720575940632425919,
        720575940637568838, 720575940629176663, 720575940621502051,
        720575940638202345, 720575940612670570, 720575940611875570,
        720575940621754367, 720575940633143833, 720575940613601698,
        720575940630797113, 720575940639198653, 720575940639259967,
        720575940624963786, 720575940640649691, 720575940610788069,
        720575940623172843, 720575940628853239,
    ],
    # Bitter gustatory receptor neurons — 200 Hz in Sugar_and_bitter experiment.
    "bitter_grn": [
        720575940619072513, 720575940646212996, 720575940622298631,
        720575940642088333, 720575940627692048, 720575940617239197,
        720575940618682526, 720575940604714528, 720575940603266592,
        720575940604027168, 720575940619197093, 720575940610259370,
        720575940627578156, 720575940629481516, 720575940618887217,
        720575940614281266, 720575940634859188, 720575940645743412,
        720575940637742911, 720575940617094208, 720575940629416318,
        720575940630195909, 720575940615641798, 720575940638312262,
        720575940624310345, 720575940621778381, 720575940619659861,
        720575940629146711, 720575940625750105, 720575940610483162,
        720575940610481370, 720575940602353632, 720575940610773090,
        720575940617433830, 720575940628962407, 720575940626287336,
        720575940623183083, 720575940618025199, 720575940619028208,
        720575940621864060, 720575940613061118, 720575940621008895,
    ],
    # Or56a olfactory receptor neurons (aversive) — 250 Hz in P9_Or56a experiment.
    "or56a": [
        720575940659222657, 720575940641403021, 720575940624211470,
        720575940616536209, 720575940615427734, 720575940628380827,
        720575940654069409, 720575940613671330, 720575940644590116,
        720575940612972328, 720575940627318696, 720575940627805096,
        720575940632190765, 720575940633031085, 720575940634955188,
        720575940621106102, 720575940615923131, 720575940608928324,
        720575940631467591, 720575940622553420, 720575940628086607,
        720575940626357586, 720575940632041043, 720575940618946901,
        720575940616095318, 720575940626411097, 720575940634614367,
        720575940603832288, 720575940620055905, 720575940609633378,
        720575940637704676, 720575940638202852, 720575940622713578,
        720575940635705963, 720575940629830508, 720575940630257772,
        720575940619539182, 720575940612019442, 720575940639931893,
    ],
    # Full Johnston's organ population (all subtypes) — 300 Hz in P9_JO_CE_bilateral.
    # JO-EV + JO-EDC + JO-EDM + JO-EDP + JO-EVL + JO-EVM + JO-EVP + JO-CA + JO-CL + JO-CM.
    "antennal_mechano_full": [
        # JO-EV
        720575940645106376, 720575940615272415, 720575940619869120,
        720575940620257345, 720575940620382889, 720575940630834683,
        720575940632449619, 720575940634020508, 720575940605530302,
        720575940607140035, 720575940608742409, 720575940615590843,
        720575940620410177, 720575940621870618, 720575940622344170,
        720575940623298559, 720575940626042149, 720575940627379333,
        720575940630080071, 720575940632128031, 720575940632307527,
        720575940634820703,
        # JO-EDC
        720575940606154370, 720575940605919334, 720575940608884931,
        720575940616655989, 720575940620543110, 720575940622937528,
        720575940624799290, 720575940626565455, 720575940627941431,
        720575940627977457, 720575940628160617, 720575940629188251,
        720575940641921421,
        # JO-EDM
        720575940615972027, 720575940618941037, 720575940619729835,
        720575940627282279, 720575940628903247, 720575940604122982,
        720575940609486690, 720575940609541917, 720575940610018266,
        720575940611061526, 720575940611273395, 720575940611684787,
        720575940614060829, 720575940616040587, 720575940618599872,
        720575940618684481, 720575940619663239, 720575940619932654,
        720575940620919578, 720575940621218729, 720575940622271684,
        720575940622638276, 720575940623312828, 720575940625797617,
        720575940625962568, 720575940626309438, 720575940626666066,
        720575940627109991, 720575940628101126, 720575940628978450,
        720575940629055721, 720575940629650997, 720575940629985900,
        720575940630992557, 720575940637054835, 720575940637084762,
        720575940638664437, 720575940646927668, 720575940646929204,
        720575940659131009,
        # JO-EDP
        720575940609522461, 720575940610261346, 720575940613641915,
        720575940615469785, 720575940616589878, 720575940616951124,
        720575940619479979, 720575940621218985, 720575940628444667,
        720575940634634606, 720575940640753267, 720575940650244342,
        # JO-EVL
        720575940615573597, 720575940615848788, 720575940619083349,
        720575940621397417, 720575940621625597, 720575940622283912,
        720575940627049731, 720575940629022149, 720575940630122015,
        720575940630564179, 720575940633153375, 720575940637410869,
        720575940638681845, 720575940621033477, 720575940621776410,
        720575940621815690, 720575940622234211, 720575940622635817,
        720575940623897096, 720575940626148354, 720575940626540821,
        720575940628258715, 720575940629743063, 720575940630202624,
        720575940630544967, 720575940633553820, 720575940644036644,
        # JO-EVM
        720575940602132509, 720575940602506208, 720575940610759634,
        720575940614188149, 720575940615809349, 720575940615976891,
        720575940619341105, 720575940621092534, 720575940622419165,
        720575940622449388, 720575940623108134, 720575940624981436,
        720575940628192055, 720575940630059847, 720575940632767383,
        720575940639296189, 720575940645466500, 720575940611783464,
        720575940612307478, 720575940612960552, 720575940614351477,
        720575940617212134, 720575940617434086, 720575940618130334,
        720575940620249734, 720575940620940276, 720575940621010352,
        720575940621729757, 720575940623437547, 720575940624546062,
        720575940624686268, 720575940625054647, 720575940625605905,
        720575940626795909, 720575940627585688, 720575940630020111,
        720575940632175268, 720575940634073183, 720575940634891700,
        720575940637012196, 720575940637243504, 720575940639339392,
        720575940659426177,
        # JO-EVP
        720575940620444654, 720575940631866508, 720575940607853833,
        720575940611088563, 720575940612773374, 720575940613221928,
        720575940615024543, 720575940615986459, 720575940617811013,
        720575940618467195, 720575940621442224, 720575940622199977,
        720575940624915230, 720575940625559358, 720575940627104649,
        720575940627314088, 720575940633058989, 720575940636335735,
        # JO-CA
        720575940605800369, 720575940608784579, 720575940618135109,
        720575940626719101, 720575940629296185, 720575940636137591,
        720575940602720940, 720575940610079857, 720575940614427195,
        720575940616501787, 720575940617156445, 720575940625909962,
        720575940626241369, 720575940629105658, 720575940629138959,
        720575940636559534, 720575940641372661,
        # JO-CL
        720575940626135548, 720575940627751567, 720575940604753437,
        720575940613971485, 720575940614835362, 720575940623399059,
        720575940630319671, 720575940639082062,
        # JO-CM
        720575940607386307, 720575940634512992, 720575940614035485,
        720575940618901424, 720575940630070343, 720575940633443353,
        720575940635058612, 720575940637632419, 720575940625626000,
    ],
    # P9 upstream input neurons — activate to drive forward walking via P9/oDN1.
    # These are the same neurons used in the P9s_100Hz_b notebook experiment.
    "p9_input": [
        720575940627652358,  # P9 left
        720575940635872101,  # P9 right
    ],
    # Full LC4 visual looming neuron population (103 neurons) from example.ipynb.
    # Used in the P9_LC4s looming escape experiment at 200 Hz.
    "lc4_looming_full": [
        720575940605598892, 720575940611134833, 720575940612580977,
        720575940613256863, 720575940613260959, 720575940614914107,
        720575940615462587, 720575940617176321, 720575940617266722,
        720575940618807105, 720575940620795728, 720575940622108001,
        720575940624017251, 720575940625038090, 720575940625934973,
        720575940625991043, 720575940626605200, 720575940626626895,
        720575940628454522, 720575940628462340, 720575940630851036,
        720575940638496720, 720575940603637438, 720575940610522009,
        720575940612093351, 720575940612323025, 720575940612380723,
        720575940612498129, 720575940612518055, 720575940612968421,
        720575940613609484, 720575940613638041, 720575940614572742,
        720575940614582946, 720575940615053580, 720575940615127227,
        720575940615232217, 720575940615575007, 720575940616066705,
        720575940616713355, 720575940617026260, 720575940617348379,
        720575940618002644, 720575940618234704, 720575940618234715,
        720575940618266459, 720575940618267227, 720575940618275520,
        720575940618312606, 720575940618676440, 720575940618709158,
        720575940618723749, 720575940619397542, 720575940620314221,
        720575940620314612, 720575940620731380, 720575940620903551,
        720575940621145821, 720575940621522458, 720575940621753579,
        720575940622330582, 720575940622531767, 720575940622939836,
        720575940624111763, 720575940624790781, 720575940624856762,
        720575940625841351, 720575940625845447, 720575940625906702,
        720575940625932421, 720575940626553596, 720575940626916936,
        720575940627519107, 720575940628064260, 720575940628081541,
        720575940628419527, 720575940628518400, 720575940628599895,
        720575940628606713, 720575940628699560, 720575940628891863,
        720575940629753807, 720575940629964591, 720575940630154660,
        720575940630484495, 720575940630998339, 720575940631032657,
        720575940631338271, 720575940632475449, 720575940632715234,
        720575940632769180, 720575940633013355, 720575940633218863,
        720575940633580384, 720575940634517856, 720575940635835967,
        720575940636957006, 720575940638456227, 720575940639817947,
        720575940640612480, 720575940641213824, 720575940645821316,
        720575940649229433, 720575940652611745,
    ],
}

# ============================================================================
# Sensory → brain rate mapping
#
# The fly body provides contact and olfaction signals.  We convert them into
# per-neuron Poisson firing rates (Hz) that are injected into the LIF model.
# ============================================================================

BASE_RATES_HZ: dict[str, float] = {
    # 200 Hz matches the Sugar_200Hz experiment in example.ipynb.
    "sugar_grn": 200.0,
    # 200 Hz for bitter GRNs — Sugar_and_bitter experiment.
    "bitter_grn": 200.0,
    # 250 Hz for Or56a olfactory neurons — P9_Or56a experiment.
    "or56a": 250.0,
    # 100 Hz matches the P9s_100Hz_b experiment in example.ipynb.
    # P9 input neurons are the primary upstream command for forward walking.
    "p9_input": 100.0,
    # 300 Hz matches P9_JO_CE_bilateral experiment in example.ipynb (all JO subtypes).
    "antennal_mechano_full": 300.0,
    # 200 Hz for full LC4 looming neuron population (P9_LC4s experiment).
    "lc4_looming_full": 200.0,
}

# ============================================================================
# Scenario registry
#
# Each entry maps a human-readable scenario name to:
#   description  — printed at run start
#   sensory      — set of SENSORY_NEURONS keys to activate throughout the run
#
# Sensory groups not in 'sensory' for the active scenario receive zero rate.
# ============================================================================

Scenario = str  # type alias: one of the keys in SCENARIOS

SCENARIOS: dict[str, dict] = {
    # Notebook: P9s_100Hz_b
    # Direct activation of P9 upstream neurons -> oDN1 -> forward walking.
    "forward_walking": {
        "description": "P9 input neurons @ 100 Hz -> oDN1 -> forward walking",
        "sensory": {"p9_input"},
    },
    # Notebook: Sugar_200Hz
    # Taste cue on legs -> MN9 (proboscis extension) + mild forward motion.
    "sugar_feeding": {
        "description": "Labellar sugar GRNs @ 200 Hz -> MN9 proboscis extension + forward",
        "sensory": {"sugar_grn"},
    },
    # Notebook: Sugar_and_bitter
    # Competing appetitive and aversive taste -> feeding suppressed, MDN elevated.
    "sugar_and_bitter": {
        "description": "Sugar GRNs + bitter GRNs @ 200 Hz -> competitive suppression, avoidance",
        "sensory": {"sugar_grn", "bitter_grn"},
    },
    # Notebook: P9_LC4s
    # Forward walking baseline + visual looming -> MDN / Giant Fibre -> escape.
    "looming_escape": {
        "description": "P9 @ 100 Hz + LC4 looming @ 200 Hz -> MDN/Giant Fibre -> backward escape",
        "sensory": {"p9_input", "lc4_looming_full"},
    },
    # Notebook: P9_JO_CE_bilateral (embodied interpretation)
    # JO-only activation: fly is stationary, antennal dust triggers aDN1 grooming.
    # P9 is intentionally absent — a grooming fly stops walking.
    # The CPG slows to a stop as groom_suppression reduces intrinsic_freq.
    "grooming": {
        "description": "Full JO population @ 300 Hz only -> aDN1 -> grooming (walking stops)",
        "sensory": {"antennal_mechano_full"},
    },
    # Direct translation of P9_JO_CE_bilateral: walking baseline + JO dust.
    # Useful for observing grooming competing against forward motion.
    "grooming_while_walking": {
        "description": "P9 @ 100 Hz + full JO @ 300 Hz -> aDN1 grooming vs oDN1 walking (notebook exact)",
        "sensory": {"p9_input", "antennal_mechano_full"},
    },
    # Notebook: P9_Or56a
    # Forward walking + aversive olfaction -> DNa01/DNa02 asymmetry -> turning.
    "olfactory_repulsion": {
        "description": "P9 + Or56a @ 250 Hz -> DNa01/DNa02 asymmetry -> turning away",
        "sensory": {"p9_input", "or56a"},
    },
}

DEFAULT_SCENARIO: Scenario = "sugar_feeding"

# ============================================================================
# DN → body control signal mapping
#
# We convert mean spike rates (Hz) over the last tick into normalised float
# control signals in [0, 1] via a sigmoid-style saturation.
#
# These are deliberately simple linear mappings as a first approximation.
# The spike rates for the published DNs at their behavioural threshold are
# not yet empirically calibrated; the values here are chosen heuristically.
#
# SATURATION_HZ: rate at which output saturates at 1.0.
# ============================================================================

DN_SATURATION_HZ: float = 40.0  # Hz; ~saturating spike rate for most DNs


def dn_rate_to_signal(mean_hz: float) -> float:
    """Convert a mean DN spike rate (Hz) to a control signal in [0, 1]."""
    return float(np.clip(mean_hz / DN_SATURATION_HZ, 0.0, 1.0))


# ============================================================================
# Brain wrapper
# ============================================================================

class BrainModel:
    """
    Thin wrapper that loads the PyTorch LIF model (batch_size=1) and exposes
    a step_tick() method for the 15 ms sync loop.

    Only the spike counts of the registered DNs are accumulated per tick;
    all other spike tensors are discarded immediately to keep RAM flat.
    """

    def __init__(self, device: str = "cpu") -> None:
        print("Brain: loading neuron ID mappings...")
        self.flyid2i, self.i2flyid = get_hash_tables(str(path_comp))

        print("Brain: loading connectome weights (may take ~30s first run)...")
        weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=(device != "mps"))
        if device != "mps":
            weights = weights.to(device=device)

        self.num_neurons = weights.shape[0]
        self.device = device

        print(f"Brain: creating LIF model ({self.num_neurons:,} neurons)...")
        self.model = TorchModel(
            batch=1,
            size=self.num_neurons,
            dt=DT,
            params=MODEL_PARAMS,
            weights=weights,
            device=device,
        )

        # Persistent state tensors (kept alive between ticks).
        self.conductance, self.delay_buffer, self.spikes, self.v, self.refrac = (
            self.model.state_init()
        )

        # Pre-compute the tensor indices of all registered DNs.
        self.dn_indices: dict[str, list[int]] = {
            grp: [self.flyid2i[fid] for fid in ids if fid in self.flyid2i]
            for grp, ids in DESCENDING_NEURONS.items()
        }

        # Pre-compute the tensor indices of all sensory neuron groups.
        self.sensory_indices: dict[str, list[int]] = {
            grp: [self.flyid2i[fid] for fid in ids if fid in self.flyid2i]
            for grp, ids in SENSORY_NEURONS.items()
        }

        # Input rates tensor (persistent, mutated per tick).
        self.rates = torch.zeros(1, self.num_neurons, device=device)

        print("Brain: ready.")

    def set_sensory_rates(self, active_groups: set[str]) -> None:
        """
        Update the Poisson input rates tensor.

        Parameters
        ----------
        active_groups:
            Set of sensory group names (keys of SENSORY_NEURONS) that are
            currently active.  All others are set to zero.
        """
        self.rates.zero_()
        for grp, indices in self.sensory_indices.items():
            if grp in active_groups and indices:
                self.rates[0, indices] = BASE_RATES_HZ[grp]

    def step_tick(self, active_sensory: set[str]) -> dict[str, float]:
        """
        Run STEPS_PER_TICK LIF steps (= 15 ms of brain time).

        Returns
        -------
        dn_rates_hz:
            Map from DN group name to mean spike rate (Hz) over the tick.
        """
        self.set_sensory_rates(active_sensory)

        # Accumulate spike counts for each DN group only.
        dn_spike_counts: dict[str, int] = {grp: 0 for grp in self.dn_indices}

        tick_duration_ms = STEPS_PER_TICK * DT  # 15 ms

        with torch.no_grad():
            for _ in range(STEPS_PER_TICK):
                (
                    self.conductance,
                    self.delay_buffer,
                    self.spikes,
                    self.v,
                    self.refrac,
                ) = self.model(
                    self.rates,
                    self.conductance,
                    self.delay_buffer,
                    self.spikes,
                    self.v,
                    self.refrac,
                )

                # Extract DN spikes only — avoids storing the full spike tensor.
                spike_row = self.spikes[0]  # shape (num_neurons,)
                for grp, indices in self.dn_indices.items():
                    if indices:
                        dn_spike_counts[grp] += int(spike_row[indices].sum().item())

        # Convert spike counts -> mean Hz over the tick window.
        dn_rates_hz = {
            grp: (count / (tick_duration_ms / 1000.0))
            for grp, count in dn_spike_counts.items()
        }

        return dn_rates_hz


# ============================================================================
# NMF (flygym) body wrapper
# ============================================================================

def build_body(render: bool = False):
    """
    Construct and return a flygym simulation and fly entity.

    Returns (sim, obs, preprogrammed_steps) where obs is the initial
    observation dict from reset() and preprogrammed_steps is the walking
    controller (or None if unavailable).
    """
    # flygym is always available (it is imported at module level).

    # Attempt to load the preprogrammed stepping controller.
    preprogrammed_steps = None
    try:
        from flygym.examples.locomotion import PreprogrammedSteps
        preprogrammed_steps = PreprogrammedSteps()
    except ImportError:
        pass

    print("Body: initialising flygym environment...")
    terrain = FlatTerrain()

    render_mode = "human" if render else None

    # Store global state for head pitch that embodied loop can update
    global global_head_pitch
    global_head_pitch = 0.0
    
    def my_head_model(leg_joint_angles, leg_contact_forces):
        global global_head_pitch
        # Head yaw, Head pitch
        return np.array([0.0, global_head_pitch])
        
    fly = Fly(
        name="fly",
        init_pose="stretch",
        control="position",
        # Vision, olfaction and raw vision rendering are disabled until their
        # outputs are wired into the brain sensory neuron pipeline.  Enabling
        # them renders camera frames every ~2 ms of sim time that go unused,
        # adding significant CPU overhead with zero benefit.
        enable_vision=False,
        enable_olfaction=False,
        render_raw_vision=False,
        head_stabilization_model=my_head_model
    )

    if render:
        cam = Camera(
            attachment_point=fly.model.worldbody,
            camera_name="camera_left",
            play_speed=0.1,
            timestamp_text=True,
        )
        cameras = [cam]
    else:
        cameras = []

    sim = Simulation(
        flies=[fly],
        arena=terrain,
        cameras=cameras,
        timestep=DT / 1000.0,  # convert ms -> seconds (DT = 0.1 ms -> 1e-4 s)
    )
    sim.cameras = cameras

    obs, info = sim.reset()
    print("Body: ready.")
    return sim, obs, preprogrammed_steps


# ============================================================================
# Sensory bridge: NMF observation -> active sensory neuron groups
# ============================================================================

def extract_active_sensory(obs: dict) -> set[str]:
    """
    Map raw flygym observation dict to the set of active sensory groups.

    Currently handles:
    - Leg contact forces  -> sugar GRN activation (placeholder: any contact)
    - Extension of taste palps -> proboscis contact
    """
    active: set[str] = set()

    # If any of the six leg tips are in contact, activate sugar GRNs.
    # In flygym, contact forces are available as 'contact_forces'.
    # In flygym v2, the observation is a dict keyed by fly name.
    fly_obs = obs.get("fly", obs)
    contact = fly_obs.get("contact_forces", None)
    if contact is not None:
        total_contact = float(np.linalg.norm(contact))
        if total_contact > 0.01:
            active.add("sugar_grn")

    return active


# ============================================================================
# Motor bridge: DN spike rates -> flygym control signals
# ============================================================================

def dn_rates_to_control(dn_rates_hz: dict[str, float]) -> dict[str, float]:
    """
    Translate DN spike rates to a complete set of NMF body control signals.

    DN -> NMF wiring
    ----------------
    odn1 / p9         -> forward CPG frequency  (intrinsic_freq = 5 * forward)
    dna01 / dna02     -> left/right CPG asymmetry -> yaw
    mdn / giant_fibre -> reversed CPG -> backward walk  (escape trumps forward)
    adn1              -> CPG pause + head reset  (groom suppresses walking)
    mn9               -> head pitch offset  (feeding proxy; best-effort)

    Returns
    -------
    dict with keys:
        "forward"   float [0, 1]   forward velocity scale
        "backward"  float [0, 1]   escape / backward walk scale
        "turn"      float [-1, 1]  yaw rate (left < 0 < right)
        "groom"     float [0, 1]   antennal grooming activation
        "feed"      float [0, 1]   proboscis extension activation (via head pitch)
    """
    p9_hz = dn_rates_hz.get("p9", 0.0)
    odn1_hz = dn_rates_hz.get("odn1", 0.0)
    # Strongest forward command wins.
    forward = dn_rate_to_signal(max(p9_hz, odn1_hz))

    dna01_sig = dn_rate_to_signal(dn_rates_hz.get("dna01", 0.0))
    dna02_sig = dn_rate_to_signal(dn_rates_hz.get("dna02", 0.0))
    turn = float(np.clip(dna01_sig - dna02_sig, -1.0, 1.0))

    groom = dn_rate_to_signal(dn_rates_hz.get("adn1", 0.0))
    feed = dn_rate_to_signal(dn_rates_hz.get("mn9", 0.0))

    # Escape: MDN drives backward walking; Giant Fibre is a fast-escape pathway.
    # The stronger of the two sets the backward signal.
    mdn_hz = dn_rates_hz.get("mdn", 0.0)
    gf_hz = dn_rates_hz.get("giant_fibre", 0.0)
    backward = dn_rate_to_signal(max(mdn_hz, gf_hz))

    return {
        "forward": forward,
        "backward": backward,
        "turn": turn,
        "groom": groom,
        "feed": feed,
    }


# ============================================================================
# Main sync loop
# ============================================================================

def select_device() -> str:
    """Return the best available torch device string."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_embodied_fly(
    duration_sec: float = 5.0,
    render: bool = False,
    device: str | None = None,
    scenario: Scenario = DEFAULT_SCENARIO,
) -> list[dict]:
    """
    Run the embodied fly for *duration_sec* simulated seconds under a named
    scenario from the SCENARIOS registry.

    Each scenario configures which sensory neuron groups fire throughout the
    run, driving the brain which in turn activates descending neurons that
    are wired to NMF body controllers.

    Returns a list of per-tick log dicts containing simulated time, DN rates,
    and control signals.
    """
    device = device or select_device()
    scenario_cfg = SCENARIOS[scenario]
    print(f"\nScenario   : {scenario}")
    print(f"Description: {scenario_cfg['description']}")
    print(f"Device     : {device.upper()}")

    # ---- Initialise subsystems ----
    # Body must be initialised AND stepped at least once BEFORE brain so
    # MuJoCo claims the OpenMP thread pool (including vision camera render)
    # before PyTorch/MPS initialises.  Without this, the MPS backend causes
    # a segfault when MuJoCo renders the eye cameras for the first time.
    sim, obs, preprogrammed_steps = build_body(render=render)

    if sim is not None:
        # Force MuJoCo to render the vision cameras on this dummy step so that
        # it acquires the OpenMP runtime before PyTorch loads.
        _zero_action = {"fly": {"joints": np.zeros(42, dtype=np.float32)}}
        obs, *_ = sim.step(_zero_action)

    brain = BrainModel(device=device)

    body_available = sim is not None

    # CPG phase/magnitude state for walking gait (6 legs).
    cpg_phases = np.zeros(6)
    cpg_magnitudes = np.ones(6)
    # Independent phase for grooming foreleg sweep (advances only when groom > 0).
    groom_phase: float = 0.0

    total_ticks = int((duration_sec * 1000.0) / TICK_MS)
    print(f"\nRunning {total_ticks} ticks x {TICK_MS} ms = {duration_sec}s simulated time")
    print("-" * 60)

    tick_log: list[dict] = []
    wall_start = time()

    for tick_i in range(total_ticks):
        sim_time_ms = tick_i * TICK_MS

        # --- Sensory phase ---
        # Start from the body-derived sensory events (contact, olfaction, etc.),
        # then merge in the scenario-mandated sensory groups.
        active_sensory: set[str] = (
            extract_active_sensory(obs) if body_available else set()
        )
        active_sensory.update(scenario_cfg["sensory"])

        # --- Brain phase ---
        dn_rates_hz = brain.step_tick(active_sensory)

        # --- Motor translation ---
        # DN spike rates are converted to NMF body control signals.
        control = dn_rates_to_control(dn_rates_hz)

        # --- Body phase ---
        if body_available:
            # Step the physics engine for STEPS_PER_TICK timesteps.
            # We regenerate the action each sub-step so the CPG advances.
            for _sub_step in range(STEPS_PER_TICK):
                action, cpg_phases, cpg_magnitudes, groom_phase = _build_flygym_action(
                    sim, control, obs, cpg_phases, cpg_magnitudes,
                    preprogrammed_steps, groom_phase,
                )
                obs, _reward, terminated, truncated, _info = sim.step(action)
                if terminated or truncated:
                    print(f"Environment ended at tick {tick_i}.")
                    body_available = False
                    break

        if render and body_available:
            sim.render()

        # --- Logging ---
        log_entry = {
            "tick": tick_i,
            "sim_time_ms": sim_time_ms,
            "active_sensory": sorted(active_sensory),
            "dn_rates_hz": dn_rates_hz,
            "control": control,
        }
        tick_log.append(log_entry)

        if (tick_i + 1) % max(1, total_ticks // 10) == 0:
            elapsed = time() - wall_start
            pct = (tick_i + 1) / total_ticks * 100
            p9_hz = dn_rates_hz.get("p9", 0.0)
            odn1_hz = dn_rates_hz.get("odn1", 0.0)
            mdn_hz = dn_rates_hz.get("mdn", 0.0)
            gf_hz = dn_rates_hz.get("giant_fibre", 0.0)
            print(
                f"  Tick {tick_i + 1:>5}/{total_ticks}  "
                f"({pct:>5.1f}%)  elapsed: {elapsed:>6.1f}s  "
                f"sim: {sim_time_ms / 1000:.2f}s  "
                f"| P9:{p9_hz:>6.1f}Hz  oDN1:{odn1_hz:>6.1f}Hz  "
                f"MN9:{dn_rates_hz.get('mn9', 0.0):>6.1f}Hz  "
                f"aDN1:{dn_rates_hz.get('adn1', 0.0):>6.1f}Hz  "
                f"MDN:{mdn_hz:>6.1f}Hz  GF:{gf_hz:>6.1f}Hz  "
                f"| fwd:{control['forward']:.2f}  "
                f"bwd:{control['backward']:.2f}  "
                f"turn:{control['turn']:+.2f}  "
                f"groom:{control['groom']:.2f}  "
                f"feed:{control['feed']:.2f}"
            )

    wall_total = time() - wall_start
    simulated_sec = total_ticks * TICK_MS / 1000.0
    print("-" * 60)
    print(f"Done.  Wall time: {wall_total:.1f}s  Sim time: {simulated_sec:.2f}s")
    print(f"       Sim/wall ratio: {simulated_sec / wall_total:.3f}x")

    if render and body_available and hasattr(sim, "cameras") and sim.cameras:
        print("Saving video to embodied_fly.mp4...")
        sim.cameras[0].save_video("embodied_fly.mp4")

    return tick_log


def _build_flygym_action(
    sim,
    control: dict,
    obs: dict,
    cpg_phases: np.ndarray,
    cpg_magnitudes: np.ndarray,
    preprogrammed_steps,
    groom_phase: float = 0.0,
) -> tuple:
    """
    Translate a high-level control dict into a flygym action array.

    DN -> NMF mapping applied here
    --------------------------------
    odn1 / p9         : effective_forward -> intrinsic CPG frequency (0-5 Hz)
    mdn / giant_fibre : backward subtracts from forward; net negative = CPG reversal
    dna01 / dna02     : turn biases CPG frequency left vs right for yaw
    adn1              : groom scales down intrinsic_freq (grooming pauses walking)
    mn9               : feed drives head pitch offset (best-effort proboscis proxy)

    flygym (control='position') expects joint angle targets for all 42 DoF
    across six legs plus per-leg adhesion signals.

    Returns (action, updated_cpg_phases, updated_cpg_magnitudes), where action
    is wrapped in a dict keyed by the fly name for flygym v2.
    """
    fly_obs = obs.get("fly", obs)

    # Net effective forward drive: escape signal subtracts from forward.
    # If backward wins, effective_forward goes negative -> reversal direction.
    effective_forward = control["forward"] - control["backward"]

    # Grooming suppresses walking: scale down CPG proportionally to groom signal.
    groom_suppression = 1.0 - control["groom"]
    intrinsic_freq = 5.0 * effective_forward * groom_suppression

    # PreprogrammedSteps leg order: [LF, LM, LH, RF, RM, RH]
    LEFT_LEGS = np.array([0, 1, 2])
    RIGHT_LEGS = np.array([3, 4, 5])

    # Turning asymmetry: positive turn biases right legs faster (turns left).
    freq_left = intrinsic_freq + max(0.0, -control["turn"]) * 2.0
    freq_right = intrinsic_freq + max(0.0, control["turn"]) * 2.0

    delta = np.zeros(6)
    delta[LEFT_LEGS] = 2.0 * np.pi * freq_left * (DT / 1000.0)
    delta[RIGHT_LEGS] = 2.0 * np.pi * freq_right * (DT / 1000.0)
    updated_phases = (cpg_phases + delta) % (2.0 * np.pi)
    updated_magnitudes = cpg_magnitudes  # held constant for now

    # Build joint-angle action via PreprogrammedSteps if available.
    # Foreleg grooming oscillation frequency (Hz) — biological grooming rate.
    GROOM_FREQ_HZ = 3.0
    GROOM_THRESHOLD = 0.3
    groom = control["groom"]

    # Advance groom_phase regardless of whether grooming is currently active so
    # it starts mid-swing when grooming kicks in (no discontinuity).
    updated_groom_phase = (
        groom_phase + 2.0 * np.pi * GROOM_FREQ_HZ * (DT / 1000.0)
    ) % (2.0 * np.pi)

    if preprogrammed_steps is not None:
        joints_angles = []
        adhesion = []
        for i, leg in enumerate(preprogrammed_steps.legs):
            angles = preprogrammed_steps.get_joint_angles(
                leg, updated_phases[i], updated_magnitudes[i]
            )
            joints_angles.append(angles)
            adhesion.append(
                preprogrammed_steps.get_adhesion_onoff(leg, updated_phases[i])
            )

        # ---- Foreleg grooming motor pattern ----
        # When aDN1 is active, blend the two foreleg joint angle arrays from
        # their walking/standing positions toward grooming target positions.
        # The blend weight is `groom` (already in [0, 1]), so transitions are
        # always smooth and physically gradual — no sudden joint jumps.
        #
        # Joint layout (7 per leg, confirmed from PreprogrammedSteps output):
        #   [0] ThC_pitch  [1] ThC_roll  [2] ThC_yaw
        #   [3] CTr_pitch  [4] CTr_roll
        #   [5] FTi_pitch  [6] TiTa_pitch
        #
        # Standing (phase=0): LF [0.61, 1.32, -0.11, -2.61, 0.92, 2.10, -0.42]
        #                      RF [0.61,-1.32,  0.11, -2.61,-0.92, 2.10, -0.42]
        #
        # Grooming targets: leg raises partially (CTr_pitch from -2.61 to -1.5),
        # ThC_yaw rotates toward midline, FTi_pitch oscillates for the sweep.
        # CTr_pitch limited to -1.5 (not -0.5) to stay within stable limits.
# ---- Foreleg grooming motor pattern ----
        if len(joints_angles[0]) == 7:
            # Anti-phase sweeps to prevent clamping and simulate rubbing
            sweep_lf = np.sin(updated_groom_phase)
            sweep_rf = np.sin(updated_groom_phase + np.pi) 

            # --- LF foreleg grooming target ---
            lf_groom = np.array([
                0.4,                        # [0] ThC_pitch: slight forward tilt
                1.321,                      # [1] ThC_roll: unchanged
                -0.45 + 0.1 * sweep_lf,     # [2] ThC_yaw: widened to prevent midline collision
                -1.5 + 0.15 * sweep_lf,     # [3] CTr_pitch: partial raise
                0.5,                        # [4] CTr_roll: stable
                2.1 + 0.3 * sweep_lf,       # [5] FTi_pitch: sweeping oscillation
                -0.3,                       # [6] TiTa_pitch: relaxed
            ])
            # Blend: groom=0 -> walking angles, groom=1 -> grooming target
            joints_angles[0] = (
                (1.0 - groom) * joints_angles[0] + groom * lf_groom
            )

            # --- RF foreleg grooming target (ThC_roll / ThC_yaw / CTr_roll mirrored) ---
            rf_groom = np.array([
                0.4,
                -1.321,
                0.45 - 0.1 * sweep_rf,      # [2] ThC_yaw: widened to prevent midline collision
                -1.5 + 0.15 * sweep_rf,     # [3] CTr_pitch: anti-phase sweep
                -0.5,                       # [4] CTr_roll: mirrored
                2.1 + 0.3 * sweep_rf,       # [5] FTi_pitch: anti-phase sweep
                -0.3,
            ])
            joints_angles[3] = (
                (1.0 - groom) * joints_angles[3] + groom * rf_groom
            )

        # Head pitch: feeding proxy (mn9). Suppressed during grooming.
        global global_head_pitch
        global_head_pitch = 0.5 * control["feed"] * (1.0 - groom)
        action = {
            "joints": np.concatenate(joints_angles),
            "adhesion": np.array(adhesion),
        }
    else:
        # Fallback: hold the current observed joint positions (no movement).
        joint_obs = fly_obs.get("joints", None)
        if joint_obs is not None and len(joint_obs) > 0:
            action = {"joints": joint_obs[0].astype(np.float32)}
        else:
            try:
                action = {
                    "joints": np.zeros(
                        sim.action_space["fly"]["joints"].shape,
                        dtype=np.float32,
                    )
                }
            except (KeyError, TypeError):
                action = {"joints": np.zeros(42, dtype=np.float32)}

    return {"fly": action}, updated_phases, updated_magnitudes, updated_groom_phase

# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embodied fly: connect the LIF brain to a flygym body."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulated duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Enable flygym visual rendering.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Torch device (default: auto-detect).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=list(SCENARIOS.keys()),
        help=(
            f"Sensory scenario to run (default: {DEFAULT_SCENARIO!r}).\n"
            + "\n".join(
                f"  {k}: {v['description']}" for k, v in SCENARIOS.items()
            )
        ),
    )
    args = parser.parse_args()

    log = run_embodied_fly(
        duration_sec=args.duration,
        render=args.render,
        device=args.device,
        scenario=args.scenario,
    )

    print(f"\nLog entries: {len(log)}")
    if log:
        last = log[-1]
        print("Last tick DN rates (Hz):")
        for grp, hz in last["dn_rates_hz"].items():
            print(f"  {grp:<14}: {hz:.2f} Hz")
        print("Last tick control signals:")
        for signal, val in last["control"].items():
            print(f"  {signal:<10}: {val:.3f}")


if __name__ == "__main__":
    main()
