"""
SeasFire Dataset — PyTorch Dataset for local/global/OCI patch-based wildfire forecasting.

Adapted from the ORORA project (Orion AI Lab, National Observatory of Athens).
"""

import math
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
from zarr.storage import ZipStore

def _open_zarr(path):
    """Open a Zarr store from a .zip or a directory."""
    if path.endswith(".zip"):
        store = ZipStore(path, mode="r")
    else:
        store = path
    try:
        return xr.open_zarr(store, consolidated=True)
    except Exception:
        return xr.open_zarr(store, consolidated=False)


# ---------------------------------------------------------------------------
# Latitude-based area weights
# ---------------------------------------------------------------------------

def normalized_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """Compute normalised latitude weights (proportional to grid-cell area)."""
    latitude = data.coords["latitude"]
    if np.any(np.isclose(np.abs(latitude), 90.0)):
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = _weight_for_latitude_vector_without_poles(latitude)
    return weights / weights.mean(skipna=False)


def _weight_for_latitude_vector_without_poles(latitude):
    delta = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if not np.isclose(np.max(latitude), 90 - delta / 2) or not np.isclose(
        np.min(latitude), -90 + delta / 2
    ):
        raise ValueError(f"Latitude vector does not start/end at ±(90 - Δ/2).")
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    delta = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if not np.isclose(np.max(latitude), 90.0) or not np.isclose(
        np.min(latitude), -90.0
    ):
        raise ValueError(f"Latitude vector does not start/end at ±90°.")
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta / 2))
    weights[[0, -1]] = np.sin(np.deg2rad(delta / 4)) ** 2
    return weights


def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f"Vector is not uniformly spaced.")
    return diff[0]


# ---------------------------------------------------------------------------
# Grid patching utilities
# ---------------------------------------------------------------------------

def split_grid(h: int, w: int, patch_h: int, patch_w: int) -> list:
    """Split a (h, w) grid into non-overlapping patches."""
    patches = []
    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            patches.append((i, j, i + patch_h, j + patch_w))
    return patches


def filter_patches_by_mask(patches, mask):
    """Keep only patches where at least one pixel passes the mask."""
    return [p for p in patches if np.any(mask[p[0] : p[2], p[1] : p[3]])]


def cross_join_patches_with_time(patches, times):
    """Cartesian product of patches × time indices."""
    return [(patch, t) for patch in patches for t in times]


# ---------------------------------------------------------------------------
# Lead-time sampling
# ---------------------------------------------------------------------------

def calculate_choice_probs(lam, temperature, num_choices):
    """Exponential-decay probabilities for lead-time sampling."""
    raw = [
        (lam / temperature) * math.e ** (-(lam * x / temperature))
        for x in np.arange(1, num_choices + 1)
    ]
    total = sum(raw)
    return [x / total for x in raw]


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_datasets(
    ds_path,
    global_ds_path,
    input_vars,
    log_transform_vars,
    oci_vars,
    clip_vars,
    keep_vars=None,
    selected_years=None,
):
    """
    Load local + global datacubes, normalise, and create OCI dataset.

    Parameters
    ----------
    ds_path : str
        Path to the 0.25° (or local-resolution) Zarr store / ZIP.
    global_ds_path : str
        Path to the 1° (global-resolution) Zarr store / ZIP.
    input_vars : list[str]
        Variables used as model inputs.
    log_transform_vars : list[str]
        Variables to log(x+1)-transform before normalisation.
    oci_vars : list[str]
        Ocean-climate index variables.
    clip_vars : dict[str, list]
        Variables to clip: {var_name: [min, max]}.
    keep_vars : list[str] or None
        Extra variables to keep in the dataset.
    selected_years : list[int] or None
        Subset of years to load. None = all.

    Returns
    -------
    ds, global_ds, oci_ds : xr.Dataset
    """
    if keep_vars is None:
        keep_vars = ["gwis_ba", "gfed_region", "ndvi", "pop_dens", "lsm", "area"]

    print(f"Loading local dataset from {ds_path} ...")
    ds = _open_zarr(ds_path)
    print(f"Loading global dataset from {global_ds_path} ...")
    global_ds = _open_zarr(global_ds_path)

    # Fix SST artefact
    if "sst" in ds:
        ds["sst"] = ds["sst"].where(ds["sst"] >= 0)
    if "sst" in global_ds:
        global_ds["sst"] = global_ds["sst"].where(global_ds["sst"] >= 0)

    # Year filtering
    if selected_years is not None:
        ds = ds.sel(time=ds.time.dt.year.isin(selected_years))
        global_ds = global_ds.sel(time=global_ds.time.dt.year.isin(selected_years))

    # Clipping
    if clip_vars:
        for var, (lo, hi) in clip_vars.items():
            ds[var] = ds[var].clip(min=lo, max=hi)
            global_ds[var] = global_ds[var].clip(min=lo, max=hi)
            
    # Log transform
    for var in log_transform_vars:
        ds[var] = np.log(ds[var] + 1)
        global_ds[var] = np.log(global_ds[var] + 1)

    # Z-score normalisation of input vars
    for var in input_vars:
        mean_val = ds[var].mean()
        std_val = ds[var].std()
        ds[var] = (ds[var] - mean_val) / std_val
        global_mean = global_ds[var].mean()
        global_std = global_ds[var].std()
        global_ds[var] = (global_ds[var] - global_mean) / global_std

    # Keep only needed variables
    vars_to_keep = list(set(keep_vars + input_vars + oci_vars))
    vars_to_keep = [v for v in vars_to_keep if v in ds.data_vars]

    print("Loading datasets into memory ...")
    ds = ds[vars_to_keep].load()
    global_ds = global_ds[[v for v in vars_to_keep if v in global_ds.data_vars]].load()

    # OCI dataset (resampled monthly, z-scored)
    oci_ds = xr.Dataset()
    for var in oci_vars:
        resampled = ds[var].fillna(0).resample(time="1ME").mean(dim="time")
        oci_ds[var] = (resampled - resampled.mean()) / resampled.std()
    oci_ds.load()

    # Positional encoding
    ds = _add_positional_vars(ds)
    global_ds = _add_positional_vars(global_ds)

    return ds, global_ds, oci_ds


def _add_positional_vars(ds):
    """Add cos/sin latitude/longitude encodings."""
    lon = ds.longitude.values
    lat = ds.latitude.values
    lon2d = np.tile(lon[np.newaxis, :], (lat.size, 1))
    lat2d = np.tile(lat[:, np.newaxis], (1, lon.size))
    ds["cos_lon"] = (("latitude", "longitude"), np.cos(lon2d * np.pi / 180))
    ds["cos_lat"] = (("latitude", "longitude"), np.cos(lat2d * np.pi / 180))
    ds["sin_lon"] = (("latitude", "longitude"), np.sin(lon2d * np.pi / 180))
    ds["sin_lat"] = (("latitude", "longitude"), np.sin(lat2d * np.pi / 180))
    return ds


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LocalGlobalOciDataset(Dataset):
    """
    Patch-based dataset that yields local, global, and OCI inputs
    for wildfire burned-area forecasting.

    Each sample is a spatial patch at a given time step, with configurable
    lag (history length) and lead time (forecast horizon).
    """

    def __init__(
        self,
        ds_local,
        ds_global,
        ds_oci,
        input_vars,
        positional_vars,
        oci_vars,
        target,
        min_lead_time,
        max_lead_time,
        local_lag,
        oci_lag,
        global_lag,
        patches,
        task="classification",
        nanfill=-1.0,
        patch_local_shape=None,
        patch_global_shape=None,
    ):
        self.task = task
        assert task in ("classification", "regression")
        self.positional_vars = positional_vars
        self.patches = patches
        self.target = target
        self.input_vars = input_vars
        self.oci_vars = oci_vars
        self.ds_local = ds_local
        self.ds_oci = ds_oci
        self.ds_global = ds_global
        self.nanfill = nanfill
        self.local_lag = local_lag
        self.oci_lag = oci_lag
        self.global_lag = global_lag
        self.min_lead_time = min_lead_time
        self.max_lead_time = max_lead_time

        # Lead-time sampling distribution
        self.temperature = 1
        self.num_choices = max_lead_time - min_lead_time + 1
        self.l = np.log(10) / self.num_choices
        self.probs = calculate_choice_probs(self.l, self.temperature, self.num_choices)

        self.patch_local_shape = (
            list(patch_local_shape) if patch_local_shape and len(patch_local_shape) > 2 else [1, 16, 16]
        )
        self.patch_global_shape = (
            list(patch_global_shape) if patch_global_shape and len(patch_global_shape) > 2 else [1, 60, 60]
        )

        self.ds_local["normalized_weights"] = normalized_latitude_weights(
            self.ds_local
        ) * xr.ones_like(self.ds_local["area"])

    def __len__(self):
        return len(self.patches)

    def update_temperature(self, increment=1.0, factor=1.0):
        self.temperature += increment
        self.temperature *= factor
        self.probs = calculate_choice_probs(self.l, self.temperature, self.num_choices)

    def _stack_time_vars(self, ds, var_names):
        """Stack variables → (C, T, H, W), broadcasting static vars over time."""
        T = ds.sizes.get("time", 1)
        arrays = []
        for var in var_names:
            da = ds[var]
            if "time" in da.dims:
                arr = da.values
            else:
                arr = np.broadcast_to(da.values, (T, *da.values.shape))
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def __getitem__(self, idx):
        (start_h, start_w, end_h, end_w), time_idx = self.patches[idx]

        # Sample lead time
        if self.min_lead_time == self.max_lead_time:
            lead_time = self.min_lead_time
        else:
            lead_time = np.random.choice(
                np.arange(self.min_lead_time, self.max_lead_time + 1), 1, p=self.probs
            )

        # --- Local input ---
        local_slice = self.ds_local.isel(
            longitude=slice(start_w, end_w),
            latitude=slice(start_h, end_h),
            time=slice(time_idx - self.local_lag - lead_time + 1, time_idx - lead_time + 1),
        )
        local_input = self._stack_time_vars(local_slice, self.input_vars)
        local_input = np.nan_to_num(local_input, nan=self.nanfill)

        # --- Local target ---
        local_target = self.ds_local.isel(
            longitude=slice(start_w, end_w),
            latitude=slice(start_h, end_h),
            time=time_idx,
        )[self.target].values
        local_target = np.squeeze(np.nan_to_num(local_target, nan=0))

        # --- Local positional ---
        local_pos = np.stack(
            [local_slice[var].values for var in self.positional_vars], axis=0
        )

        # --- Local mask ---
        local_mask = np.isnan(local_slice.isel(time=-1)["ndvi"]).values

        # --- OCI input ---
        oci_slice = self.ds_oci.sel(
            time=slice(
                local_slice.time[-1] - np.timedelta64(self.oci_lag * 31, "D"),
                local_slice.time[-1],
            )
        )
        oci_input = oci_slice.isel(time=slice(-self.oci_lag, None))[self.oci_vars]
        oci_input = np.stack([oci_input[var] for var in self.oci_vars], axis=0)
        oci_input = np.nan_to_num(oci_input, nan=self.nanfill)

        # --- Global input ---
        global_slice = self.ds_global.isel(
            time=slice(time_idx - self.global_lag - lead_time + 1, time_idx - lead_time + 1)
        )
        global_input = self._stack_time_vars(global_slice, self.input_vars)
        global_input = np.nan_to_num(global_input, nan=self.nanfill)

        global_target = np.squeeze(
            self.ds_global.isel(time=time_idx)[self.target].values
        )
        global_target = np.nan_to_num(global_target, nan=0)

        global_pos = np.stack(
            [self.ds_global[var].values for var in self.positional_vars], axis=0
        )

        # --- Task-specific target transform ---
        if self.task == "classification":
            local_target = np.where(local_target > 0, 1, 0)
            global_target = np.where(global_target > 0, 1, 0)

        return {
            "x_local": local_input,
            "x_local_mask": local_mask,
            "x_local_pos": local_pos,
            "x_oci": oci_input,
            "x_global": global_input,
            "x_global_pos": global_pos,
            "y_local": local_target,
            "y_global": global_target,
            "lead_time": lead_time,
            "normalized_weights": local_slice["normalized_weights"].values,
        }


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def create_dataset_for_years(
    ds,
    global_ds,
    oci_ds,
    input_vars,
    oci_vars,
    positional_vars,
    target,
    min_lead_time,
    max_lead_time,
    local_lag,
    oci_lag,
    global_lag,
    patch_size,
    years,
    task="classification",
    nanfill=-1.0,
    patch_local_shape=None,
    patch_global_shape=None,
):
    """Create a LocalGlobalOciDataset for the given years."""
    patch_h, patch_w = patch_size[-2], patch_size[-1]
    patches = split_grid(ds.latitude.size, ds.longitude.size, patch_h, patch_w)

    # Keep only patches that intersect a GFED region
    mask = ds.gfed_region > 0
    patches = filter_patches_by_mask(patches, mask)

    # Time indices for requested years
    time_indices = [
        i for i, t in enumerate(ds.time) if t.dt.year.item() in years
    ]

    patches_times = cross_join_patches_with_time(patches, time_indices)

    return LocalGlobalOciDataset(
        ds,
        global_ds,
        oci_ds,
        input_vars,
        positional_vars,
        oci_vars,
        target,
        min_lead_time,
        max_lead_time,
        local_lag,
        oci_lag,
        global_lag,
        patches_times,
        task=task,
        nanfill=nanfill,
        patch_local_shape=patch_local_shape,
        patch_global_shape=patch_global_shape,
    )
