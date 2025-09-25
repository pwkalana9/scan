from dataclasses import dataclass

@dataclass
class RadarDims:
    A: int = 22
    R: int = 100
    D: int = 64
    T: int = 10

@dataclass
class RadarPhys:
    lambda_m: float = 0.031   # ~X-band
    v_max_mps: float = 50.0
    az_fov_deg: float = 120.0
    r_max: float = 2000.0

@dataclass
class Shadows:
    decay_az: float = 2.5
    decay_r: float = 4.0

