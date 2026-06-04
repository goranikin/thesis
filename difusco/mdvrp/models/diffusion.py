"""
Re-export the binary categorical diffusion used by CVRP. MDVRP's
customer-depot edges are also binary (assigned / not), so the diffusion
process is unchanged.
"""

from difusco.cvrp.models.diffusion import CategoricalDiffusion, InferenceSchedule

__all__ = ["CategoricalDiffusion", "InferenceSchedule"]
