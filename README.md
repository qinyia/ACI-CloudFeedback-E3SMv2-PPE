# ACI-CloudFeedback-E3SMv2-PPE
Scripts and data to accompany "Impact of Turbulence on the Relationship between Cloud Feedback and Aerosol-Cloud Interaction in an E3SMv2 Perturbed Parameter Ensemble" by Qin et al (2024)

## Usage
- General utility functions: `utils_PPE.py`,`utils_v1v2.py`
- Calculate ERFaci decomposition following Gryspeerdt et al. (2020): `aerocom_forcing_fraction_minimal_ppe.py`
- Calculate cloud feedback, ERFaci in partitioned regimes: `calc_ppe_fbk_aci_cloud_regime2_regime-avg.py`
- Calculate state variables in partitioned regimes: `calc_ppe_state_var_cloud_regime2_regime-avg.py`
- Calculate ERFaci decomposition in partitioned regimes: `calc_ppe_GryspeerdtDecomp_cloud_regime2_regime-avg.py`
- Generate all figures: `plot_ppe_cloud_regime_new.ipynb`

## Reference
Gryspeerdt, E., Mülmenstädt, J., Gettelman, A., Malavelle, F. F., Morrison, H., Neubauer, D., Partridge, D. G., Stier, P., Takemura, T., Wang, H., Wang, M., & Zhang, K. (2020). Surprising similarities in model and observational aerosol radiative forcing estimates. Atmospheric Chemistry and Physics, 20(1), 613–623. https://doi.org/10.5194/acp-20-613-2020
