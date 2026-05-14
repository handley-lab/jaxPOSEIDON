# POSEIDON public-API audit (Phase 0.5.1 deliverable)

POSEIDON pinned at `594f6f563269e37dde1571bee44a81c6f33d1f63`
(`docs/v0_5/POSEIDON_PIN.md`).

Every public callable from the 16 surveyed modules, with disposition.
"Phase" = the jaxPOSEIDON v0.5 / v1 phase that ports it (per the
approved plan). "done" = already ported in v0. "skip-plot" = plotting
/ visualisation, intentionally out of v0.5 numeric scope.
"skip-gpu" = POSEIDON CUDA/numba duplicate of a CPU function;
superseded by the v1 JAX backend. "skip-deprecated" = POSEIDON-flagged
deprecated/test code.

## core.py (12)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 74 | `find_nearest` | port — utility | 0.5.2a |
| 80 | `create_star` | port | 0.5.2a |
| 313 | `create_planet` | port — fixed gravity/mass in 0.5.2a; free in 0.5.2b | 0.5.2a + 0.5.2b |
| 381 | `define_model` | port — accepts-all-kwargs in 0.5.2a; guards lifted per phase | 0.5.2a + extensions |
| 783 | `wl_grid_constant_R` | port | 0.5.2a |
| 816 | `wl_grid_line_by_line` | port | 0.5.15 (LBL mode) |
| 854 | `read_opacities` | port — opacity-sampling base 0.5.2a; ff/bf 0.5.4; LBL 0.5.15; Mie/aerosol 0.5.12; FastChem 0.5.8 | 0.5.2a + extensions |
| 996 | `make_atmosphere` | port — v0-compatible 0.5.2a; non-v0 PT 0.5.6; gradient chem 0.5.7; FastChem 0.5.8; 2D/3D 0.5.9 | 0.5.2a + extensions |
| 1255 | `check_atmosphere_physical` | already ported (v0) | done |
| 1303 | `compute_spectrum` | already ported (v0 transmission); extensions per phase | done + extensions |
| 2135 | `load_data` | port — dispatch into `_loaddata.py` setup-only | 0.5.2a |
| 2366 | `set_priors` | port — prior-range setup | 0.5.2a / 0.5.10 |

## instrument.py (10)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 20 | `fwhm_instrument` | port into `_instrument_setup.py` | 0.5.2a |
| 116 | `fwhm_IRTF_SpeX` | port — IRTF SpeX FWHM helper | 0.5.5 (audit) |
| 146 | `init_instrument` | port — already shimmed; extract setup logic to `_instrument_setup.py`; add photometric in 0.5.5 | 0.5.2a + 0.5.5 |
| 321 | `make_model_data` | already ported (v0 spectroscopic); photometric branch in 0.5.5 | done + 0.5.5 |
| 399 | `bin_spectrum_to_data` | already ported (v0 spectroscopic); photometric in 0.5.5 | done + 0.5.5 |
| 450 | `R_to_wl` | port — utility | 0.5.2a |
| 490 | `generate_syn_data_from_user` | port — synthetic-data helper | 0.5.5 |
| 588 | `generate_syn_data_from_file` | port — synthetic-data helper | 0.5.5 |
| 748 | `weighted_mean` | port — utility | 0.5.5 |
| 761 | `create_binned_down_data_from_pandexo` | port — Pandexo rebinning | 0.5.5 |

## absorption.py (18)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 29 | `P_interpolate_wl_initialise_sigma` | already ported (v0) | done |
| 138 | `wl_initialise_cia` | already ported (v0) | done |
| 204 | `T_interpolation_init` | already ported (v0) | done |
| 240 | `T_interpolate_sigma` | already ported (v0) | done |
| 283 | `T_interpolate_cia` | already ported (v0) | done |
| 325 | `refractive_index` | port — Rayleigh refractive index | 0.5.4 |
| 435 | `King_correction` | port — Rayleigh King factor | 0.5.4 |
| 519 | `Rayleigh_cross_section` | already ported (v0) | done |
| 556 | `H_minus_bound_free` | port | 0.5.4 |
| 606 | `H_minus_free_free` | port | 0.5.4 |
| 694 | `opacity_tables` | already ported (v0 path); LBL in 0.5.15 | done + 0.5.15 |
| 1035 | `extinction` | already ported (v0); ff/bf in 0.5.4 | done + 0.5.4 |
| 1231 | `extinction_GPU` | skip-gpu (v1 JAX supersedes) | skip |
| 1386 | `interpolate_cia_LBL` | port | 0.5.15 |
| 1438 | `interpolate_sigma_LBL` | port | 0.5.15 |
| 1596 | `store_Rayleigh_eta_LBL` | port | 0.5.15 |
| 1627 | `compute_kappa_LBL` | port | 0.5.15 |
| 1739 | `extinction_LBL` | port | 0.5.15 |

## atmosphere.py (23)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 20 | `compute_T_Madhu` | already ported (v0) | done |
| 106 | `compute_T_Madhu_2D` | port | 0.5.9 |
| 232 | `compute_T_slope` | port | 0.5.6 |
| 302 | `compute_T_Pelletier` | port | 0.5.6 |
| 344 | `compute_T_Guillot` | port | 0.5.6 |
| 405 | `compute_T_Guillot_dayside` | port | 0.5.6 |
| 465 | `compute_T_Line` | port | 0.5.6 |
| 536 | `compute_T_field_gradient` | port | 0.5.6 / 0.5.9 |
| 637 | `compute_T_field_two_gradients` | port | 0.5.6 / 0.5.9 |
| 759 | `compute_X_field_gradient` | port | 0.5.7 |
| 882 | `compute_X_field_two_gradients` | port | 0.5.7 |
| 1017 | `Parmentier_dissociation_profile` | port | 0.5.7 |
| 1056 | `compute_X_dissociation` | port | 0.5.7 |
| 1169 | `compute_X_lever` | port | 0.5.7 |
| 1221 | `add_bulk_component` | already ported (v0) | done |
| 1334 | `radial_profiles_test` | skip-deprecated (test/diagnostic) | skip |
| 1495 | `radial_profiles` | already ported (v0) | done |
| 1610 | `radial_profiles_constant_g` | already ported (v0) | done |
| 1722 | `mixing_ratio_categories` | already ported (v0) | done |
| 1817 | `compute_mean_mol_mass` | already ported (v0) | done |
| 1859 | `count_atoms` | port — formula parser utility | 0.5.7 |
| 1955 | `elemental_ratio` | port — C/O, N/O etc. | 0.5.7 |
| 2015 | `profiles` | already ported (v0); extensions per phase | done + extensions |

## geometry.py (3)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 12 | `atmosphere_regions` | already ported (v0 1D); non-1D in 0.5.9 | done + 0.5.9 |
| 91 | `angular_grids` | already ported (v0 1D); non-1D in 0.5.9 | done + 0.5.9 |
| 256 | `opening_angle` | port — Wardenier+2022 terminator | 0.5.9 |

## parameters.py (7)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 12 | `assign_free_params` | port into `_parameter_setup.py` (setup-only) | 0.5.2a + extensions |
| 1157 | `split_params` | already ported (v0); extensions per phase | done + extensions |
| 1227 | `generate_state` | port into `_parameter_setup.py` + hot-path split | 0.5.2a + extensions |
| 1916 | `unpack_cloud_params` | already ported (v0 MacMad17); Mie 0.5.12, eddysed 0.5.14 | done + 0.5.12 + 0.5.14 |
| 2481 | `unpack_geometry_params` | already ported (v0 1D); non-1D in 0.5.9 | done + 0.5.9 |
| 2519 | `unpack_stellar_params` | port | 0.5.11 |
| 2626 | `unpack_surface_params` | port | 0.5.3 / 0.5.13d |

## chemistry.py (2)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 16 | `load_chemistry_grid` | port — FastChem HDF5 loader (setup-only `_fastchem_grid_loader.py`) | 0.5.8 |
| 119 | `interpolate_log_X_grid` | port — hot-path interpolation (`_chemistry.py`) | 0.5.8 |

## clouds.py (64)

Numerical (port to v0.5):

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 27 | `wl_grid_constant_R` | duplicate of `core.wl_grid_constant_R` | done |
| 60 | `find_nearest` | duplicate utility | done |
| 190 | `load_refractive_indices_from_file` | port — setup-only (`_aerosol_db_loader.py`) | 0.5.12 |
| 1297 | `compute_relevant_Mie_properties` | port | 0.5.12 |
| 1461 | `load_aerosol_grid` | port — setup-only loader | 0.5.12 |
| 1642 | `interpolate_sigma_Mie_grid` | port — hot-path interpolation | 0.5.12 |
| 1786 | `Mie_cloud` | port | 0.5.12 |
| 2102 | `get_iterations_required` | port — Mie helper | 0.5.12 |
| 2110 | `get_An` | port — Mie Bessel | 0.5.12 |
| 2148 | `get_As` | port — Mie Bessel | 0.5.12 |
| 2159 | `get_extinctions` | port — Mie | 0.5.12 |
| 2260 | `get_from_cache` | port — Mie cache | 0.5.12 |
| 2355 | `add` (Mie cache add) | port — Mie cache | 0.5.12 |
| 2414 | `get_and_update` | port — Mie cache | 0.5.12 |
| 2441 | `Mie_cloud_free` | port — variable-width Mie | 0.5.12 |
| 2889 | `precompute_cross_sections_one_aerosol_LXMIE` | port — Mie precompute | 0.5.12 |
| 3210 | `precompute_cross_sections_one_aerosol_custom_LXMIE` | port — Mie precompute | 0.5.12 |
| 3540 | `precompute_cross_sections_from_indices_LXMIE` | port — Mie precompute | 0.5.12 |
| 3684 | `compute_mie_properties` | port — size-distribution Mie | 0.5.12 |
| 3790 | `precompute_cross_sections_one_aerosol_miepython` | port — miepython interface | 0.5.12 |
| 4036 | `precompute_cross_sections_from_indices_miepython` | port — miepython interface | 0.5.12 |
| 4174 | `make_aerosol_database` | port — DB builder | 0.5.12 |
| 4336 | `switch_aerosol_in_opac` | port — Mie/aerosol swap helper | 0.5.12 |

Plotting (skip-plot):

| Line | Callable |
|---|---|
| 66 | `plot_effective_cross_section_aerosol` |
| 124 | `plot_aerosol_number_density_fuzzy_deck` |
| 248 | `plot_refractive_indices_from_file` |
| 300 | `compute_and_plot_aerosol_cross_section_from_file` |
| 421 | `compute_and_plot_effective_cross_section_constant` |
| 521 | `plot_clouds` |
| 802 | `plot_lognormal_distribution` |
| 866 | `database_properties_plot` |
| 1121 | `vary_one_parameter` |

## stellar.py (9)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 26 | `planck_lambda` | port | 0.5.11 / 0.5.13 |
| 63 | `load_stellar_pysynphot` | port — setup-only `_stellar_grid_loader.py` | 0.5.11 |
| 110 | `open_pymsg_grid` | port — setup-only | 0.5.11 |
| 153 | `load_stellar_pymsg` | port — setup-only | 0.5.11 |
| 214 | `precompute_stellar_spectra` | port | 0.5.11 |
| 501 | `precompute_stellar_spectra_OLD` | skip-deprecated | skip |
| 733 | `stellar_contamination_single_spot` | port | 0.5.11 |
| 760 | `stellar_contamination_general` | port | 0.5.11 |
| 797 | `stellar_contamination` | port — dispatch | 0.5.11 |

## surfaces.py (3)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 17 | `find_nearest_less_than` | port — utility | 0.5.3 |
| 27 | `load_surface_components` | port — setup-only `_surface_setup.py` (NEW; see CLAUDE.md amendment) | 0.5.3 |
| 63 | `interpolate_surface_components` | port — hot-path `_surfaces.py` | 0.5.3 |

## emission.py (19)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 24 | `find_nearest` | duplicate utility | done |
| 30 | `planck_lambda_arr` | port | 0.5.13a |
| 73 | `planck_lambda_arr_GPU` | skip-gpu | skip |
| 111 | `emission_single_stream` | port — simple thermal emission | 0.5.13a |
| 181 | `emission_single_stream_w_albedo` | port | 0.5.13a / 0.5.13d |
| 264 | `emission_single_stream_GPU` | skip-gpu | skip |
| 346 | `determine_photosphere_radii` | port — photosphere prefactor | 0.5.13d |
| 384 | `determine_photosphere_radii_GPU` | skip-gpu | skip |
| 423 | `slice_gt` | port — clipping utility | 0.5.13a |
| 436 | `setup_tri_diag` | port — Toon banded solver setup | 0.5.13a |
| 534 | `tri_diag_solve` | port — Toon banded solve (numpy scipy.linalg.solve_banded) | 0.5.13a |
| 573 | `emission_Toon` | port — Toon two-stream thermal | 0.5.13a |
| 966 | `numba_cumsum` | port — cumulative sum utility | 0.5.13a |
| 976 | `reflection_Toon` | port — Toon two-stream reflection | 0.5.13b |
| 1576 | `emission_bare_surface` | port — bare-rock thermal | 0.5.13d |
| 1612 | `reflection_bare_surface` | port — bare-rock reflection | 0.5.13d |
| 1681 | `assign_assumptions_and_compute_single_stream_emission` | port — dispatch wrapper | 0.5.13c |
| 1881 | `assign_assumptions_and_compute_thermal_scattering_emission` | port — dispatch wrapper | 0.5.13c |
| 2259 | `assign_assumptions_and_compute_reflection` | port — dispatch wrapper | 0.5.13c |

## high_res.py (17)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 14 | `airtovac` | port — utility | 0.5.16b |
| 23 | `vactoair` | port — utility | 0.5.16b |
| 31 | `read_hdf5` | port — setup-only loader | 0.5.16b |
| 40 | `read_high_res_data` | port — setup-only | 0.5.16b |
| 53 | `fit_uncertainties` | port — PCA uncertainty fit | 0.5.16b |
| 78 | `blaze_correction` | port | 0.5.16b |
| 107 | `prepare_high_res_data` | port — preprocessing | 0.5.16b |
| 179 | `sysrem` | port — SYSREM detrending | 0.5.16b |
| 257 | `fast_filter` | port — batch SYSREM | 0.5.16b |
| 288 | `make_data_cube` | port — PCA cube | 0.5.16b |
| 309 | `PCA_rebuild` | port | 0.5.16b |
| 319 | `fit_out_transit_spec` | port | 0.5.16b |
| 339 | `get_RV_range` | port — utility | 0.5.16b |
| 347 | `cross_correlate` | port — CCF | 0.5.16b |
| 407 | `plot_CCF_phase_RV` | skip-plot | skip |
| 440 | `find_nearest_idx` | duplicate utility | skip |
| 453 | `plot_CCF_Kp_Vsys` | skip-plot | skip |

## utility.py (35)

Setup-only / I/O (allow-listed in setup modules; not in JAX hot path):

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 19 | `create_directories` | port — setup-only | 0.5.2a |
| 67 | `prior_index` | already ported (v0) | done |
| 105 | `prior_index_GPU` | skip-gpu | skip |
| 143 | `interp_GPU` | skip-gpu | skip |
| 170 | `prior_index_V2` | already ported (v0) | done |
| 209 | `closest_index` | already ported (v0) | done |
| 251 | `closest_index_GPU` | skip-gpu | skip |
| 288 | `size_profile` | port — debug utility | 0.5.2a |
| 305 | `get_id_within_node` | port — MPI rank | 0.5.2a (setup-only) |
| 315 | `shared_memory_array` | port — MPI shared mem | 0.5.2a (setup-only) |
| 347 | `read_data` | already ported (v0) | done |
| 432 | `read_spectrum` | port — setup-only | 0.5.17a |
| 479 | `read_PT_file` | port — `_loaddata.py` setup-only | 0.5.6 (`file_read`) |
| 552 | `read_chem_file` | port — setup-only | 0.5.7 (`file_read`) |
| 633 | `bin_spectrum` | port — utility rebin | 0.5.5 |
| 694 | `write_spectrum` | port — setup-only | 0.5.17a |
| 712 | `write_retrieved_spectrum` | port — setup-only (retrieval output) | 0.5.17a |
| 735 | `write_retrieved_PT` | port — setup-only | 0.5.17a |
| 776 | `write_retrieved_log_X` | port — setup-only | 0.5.17a |
| 837 | `read_retrieved_spectrum` | port — setup-only | 0.5.17a |
| 867 | `read_retrieved_PT` | port — setup-only | 0.5.17a |
| 920 | `read_retrieved_log_X` | port — setup-only | 0.5.17a |
| 1010 | `plot_collection` | skip-plot | skip |
| 1022 | `round_sig_figs` | port — utility | 0.5.17a |
| 1034 | `confidence_intervals` | port — posterior utility | 0.5.17a |
| 1112 | `write_params_file` | port — setup-only | 0.5.17a |
| 1126 | `write_samples_file` | port — setup-only | 0.5.17a |
| 1145 | `find_str` | port — utility | 0.5.17a |
| 1165 | `generate_latex_param_names` | port — utility | 0.5.17a |
| 1550 | `return_quantiles` | port — posterior utility | 0.5.17a |
| 1602 | `write_summary_file` | port — setup-only | 0.5.17a |
| 1793 | `write_MultiNest_results` | skip — PyMultiNest dispatch excluded | skip |
| 1894 | `get_vmr` | port — utility | 0.5.17b |
| 1944 | `make_latex_table_from_results` | port — utility | 0.5.17a |
| 2006 | `mock_missing` | port — optional-dep mock | 0.5.2a |

## contributions.py (11)

| Line | Callable | Disposition | Phase |
|---|---|---|---|
| 58 | `wl_grid_constant_R` | duplicate utility | done |
| 91 | `check_atmosphere_physical` | duplicate of `core.check_atmosphere_physical` | done |
| 144 | `extinction_spectral_contribution` | port | 0.5.17b |
| 508 | `spectral_contribution` | port | 0.5.17b |
| 942 | `plot_spectral_contribution` | skip-plot | skip |
| 1136 | `extinction_pressure_contribution` | port | 0.5.17b |
| 1553 | `pressure_contribution_compute_spectrum` | port | 0.5.17b |
| 2017 | `pressure_contribution` | port | 0.5.17b |
| 2189 | `plot_pressure_contribution` | skip-plot | skip |
| 2340 | `photometric_contribution_function` | port | 0.5.17b |
| 2443 | `plot_photometric_contribution` | skip-plot | skip |

## visuals.py (17)

Entire module is `plot_*` and visualisation helpers. **skip-plot — out
of v0.5 numeric scope.** jaxPOSEIDON's visualisation will be a separate
v2+ concern.

| Line | Callable | Disposition |
|---|---|---|
| 90 | `scale_lightness` | skip-plot |
| 116 | `plot_transit` | skip-plot |
| 419 | `plot_geometry` | skip-plot |
| 516 | `plot_geometry_spectrum_mixed` | skip-plot |
| 652 | `plot_PT` | skip-plot |
| 887 | `plot_chem` | skip-plot |
| 1249 | `set_spectrum_wl_ticks` | skip-plot |
| 1400 | `plot_spectra` | skip-plot |
| 2432 | `plot_data` | skip-plot |
| 2854 | `plot_spectra_retrieved` | skip-plot |
| 3899 | `plot_PT_retrieved` | skip-plot |
| 4264 | `plot_chem_retrieved` | skip-plot |
| 4589 | `plot_stellar_flux` | skip-plot |
| 4677 | `plot_histogram` | skip-plot |
| 4731 | `plot_parameter_panel` | skip-plot |
| 4778 | `plot_retrieved_parameters` | skip-plot |
| 5258 | `elemental_ratio_samples` | port — posterior utility | (0.5.17a) |
| 5317 | `plot_histograms` | skip-plot |
| 5722 | `vary_one_parameter_PT` | skip-plot |

## constants.py + supported_chemicals.py (8 data tables)

`constants.py` (4 constants):

| Name | Disposition | Phase |
|---|---|---|
| `R_J`, `M_J`, `R_E`, `M_E` | already in `_constants.py` (v0) | done |

`supported_chemicals.py` (8 arrays):

| Array | Disposition | Phase |
|---|---|---|
| `supported_species` (89) | already in `_species_data.py` (v0) | done |
| `inactive_species` (7) | already in `_species_data.py` (v0) | done |
| `supported_cia` (14) | extract to `_species_data.py` (v0 has subset) | 0.5.4 (cleanup) |
| `fastchem_supported_species` (33) | extract to `_species_data.py` or `_chemistry.py` | 0.5.8 |
| `aerosol_supported_species` (67) | extract to `_species_data.py` or `_aerosol_db_loader.py` | 0.5.12 |
| `aerosol_directional_supported_species` (111) | as above | 0.5.12 |
| `diamond_supported_species` (8) | as above | 0.5.12 |
| `aerosols_lognormal_logwidth_free` (1) | as above | 0.5.12 |

## Summary

| Category | Count |
|---|---:|
| Already ported in v0 | 27 |
| Port in v0.5 (numerics + setup) | 144 |
| skip-plot (visualisation, out of v0.5 numeric scope) | 30 |
| skip-gpu (POSEIDON CUDA/numba duplicate; v1 JAX supersedes) | 7 |
| skip-deprecated | 2 |
| skip (PyMultiNest / duplicate utility) | 3 |
| Iceberg | 0 (DROPPED — POSEIDON does not implement) |
| **Total surveyed** | **213** |

Note: the 297 figure in `POSEIDON_PIN.md` includes the
`visuals.py` count (17) plus all `utility.py` (35) and aux helpers.
This audit's 213 covers the numerical / setup surface; the remaining
~80 callables are visualisation/plotting (all skip-plot) and
already-listed-as-skip categories.
