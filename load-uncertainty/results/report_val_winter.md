# Uncertainty eval report: `val_winter`

Reference (declared in advance): **val_winter__baseline__lightgbm-v2**. Negative `vs_ref_%` = lower pinball loss = better. `zone_skill_%` = mean over the 8 weather zones of each zone's % improvement vs the reference (equal weight; positive = better); the system total is reported on its own. CIs: day-clustered bootstrap. Unparseable answers are scored as p10=p50=p90=0 ("trust ERCOT exactly").

## Headline

| run                                                                              |   n |   days |   pinball_MW |   vs_ref_% | ci95_%        | P(better)   |   zone_skill_% | zone_skill_ci95   |   system_total_skill_% | below_p10   | below_p50   | below_p90   | cov_80   |   width_MW | within_1sd   | within_2sd   | beyond_3sd   |   parse_fail_% |
|:---------------------------------------------------------------------------------|----:|-------:|-------------:|-----------:|:--------------|:------------|---------------:|:------------------|-----------------------:|:------------|:------------|:------------|:---------|-----------:|:-------------|:-------------|:-------------|---------------:|
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | 810 |     90 |       112.10 |      -0.16 | [-0.5, +0.2]  | 81.4%       |           0.13 | [-0.1, +0.4]      |                   0.17 | 6.0%        | 42.0%       | 85.4%       | 79.3%    |    1192.72 | 72.0%        | 93.1%        | 1.5%         |           0.00 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | 810 |     90 |       112.27 |      -0.00 | [-0.0, +0.0]  | 93.5%       |           0.00 | [-0.0, +0.0]      |                   0.00 | 6.3%        | 42.6%       | 85.4%       | 79.1%    |    1183.20 | 71.8%        | 92.7%        | 1.6%         |           0.00 |
| val_winter__baseline__lightgbm-v2                                                | 810 |     90 |       112.28 |       0.00 | [+0.0, +0.0]  | 0.0%        |           0.00 | [+0.0, +0.0]      |                   0.00 | 6.2%        | 42.6%       | 85.4%       | 79.2%    |    1183.20 | 71.7%        | 92.7%        | 1.6%         |           0.00 |
| val_winter__baseline__lightgbm-v1                                                | 810 |     90 |       127.00 |      13.11 | [+8.5, +18.2] | 0.0%        |         -11.63 | [-17.0, -6.2]     |                 -14.06 | 7.3%        | 43.8%       | 85.3%       | 78.0%    |    1248.64 | 68.4%        | 94.5%        | 0.7%         |           0.00 |

Calibration targets: below p10 = 10%, below p50 = 50%, below p90 = 90%; within 1 sigma = 68%, within 2 sigma = 95%, beyond 3 sigma = 0.3% (sigma view reads the p10-p90 range as +/-1.28 sigma).

## Calibration by zone (share below p10 / below p90; targets 10% / 90%)

|                                                                                  | coast   | east   | far_west   | north   | north_central   | south   | south_central   | system_total   | west   |
|:---------------------------------------------------------------------------------|:--------|:-------|:-----------|:--------|:----------------|:--------|:----------------|:---------------|:-------|
| val_winter__baseline__lightgbm-v2                                                | 0%/88%  | 9%/87% | 15%/92%    | 4%/98%  | 4%/90%          | 7%/93%  | 12%/96%         | 3%/92%         | 1%/34% |
| val_winter__baseline__lightgbm-v1                                                | 1%/90%  | 7%/79% | 18%/95%    | 4%/96%  | 8%/88%          | 9%/89%  | 11%/94%         | 8%/90%         | 0%/45% |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | 0%/88%  | 8%/88% | 15%/91%    | 4%/97%  | 4%/89%          | 7%/93%  | 12%/96%         | 3%/92%         | 1%/33% |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | 0%/88%  | 9%/87% | 15%/92%    | 4%/98%  | 4%/90%          | 7%/93%  | 13%/96%         | 3%/92%         | 1%/34% |

## Slice: models_agreed (% vs reference, negative = better)

| run                                                                              |   False (n=536) |   True (n=274) |
|:---------------------------------------------------------------------------------|----------------:|---------------:|
| val_winter__baseline__lightgbm-v1                                                |          13.420 |         12.294 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.004 |         -0.004 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.197 |         -0.055 |

## Slice: big_miss (% vs reference, negative = better)

| run                                                                              |   False (n=551) |   True (n=259) |
|:---------------------------------------------------------------------------------|----------------:|---------------:|
| val_winter__baseline__lightgbm-v1                                                |          11.417 |         15.030 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.006 |         -0.014 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.574 |         -0.986 |

## Slice: theme_storms (% vs reference, negative = better)

| run                                                                              |   False (n=524) |   True (n=286) |
|:---------------------------------------------------------------------------------|----------------:|---------------:|
| val_winter__baseline__lightgbm-v1                                                |          14.733 |         11.464 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.000 |         -0.008 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.205 |         -0.110 |

## Slice: theme_severe (% vs reference, negative = better)

| run                                                                              |   False (n=734) |   True (n=76) |
|:---------------------------------------------------------------------------------|----------------:|--------------:|
| val_winter__baseline__lightgbm-v1                                                |          13.884 |         5.512 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |         0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.003 |        -0.010 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.242 |         0.672 |

## Slice: theme_heat (% vs reference, negative = better)

| run                                                                              |   False (n=711) |   True (n=99) |
|:---------------------------------------------------------------------------------|----------------:|--------------:|
| val_winter__baseline__lightgbm-v1                                                |          12.623 |        15.950 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |         0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.006 |         0.007 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.259 |         0.426 |

## Slice: theme_heavy_rain (% vs reference, negative = better)

| run                                                                              |   False (n=772) |   True (n=38) |
|:---------------------------------------------------------------------------------|----------------:|--------------:|
| val_winter__baseline__lightgbm-v1                                                |          12.508 |        19.806 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |         0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.002 |        -0.022 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.186 |         0.155 |

## Slice: theme_tropical (% vs reference, negative = better)

| run                                                                              |   False (n=794) |   True (n=16) |
|:---------------------------------------------------------------------------------|----------------:|--------------:|
| val_winter__baseline__lightgbm-v1                                                |          12.587 |        26.329 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |         0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.003 |        -0.013 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.140 |        -0.617 |

## Slice: theme_quiet (% vs reference, negative = better)

| run                                                                              |   False (n=564) |   True (n=246) |
|:---------------------------------------------------------------------------------|----------------:|---------------:|
| val_winter__baseline__lightgbm-v1                                                |          13.848 |         11.571 |
| val_winter__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.004 |         -0.004 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.366 |          0.278 |

## Slice: zone (% vs reference, negative = better)

| run                                                                              |   coast (n=90) |   east (n=90) |   far_west (n=90) |   north (n=90) |   north_central (n=90) |   south (n=90) |   south_central (n=90) |   system_total (n=90) |   west (n=90) |
|:---------------------------------------------------------------------------------|---------------:|--------------:|------------------:|---------------:|-----------------------:|---------------:|-----------------------:|----------------------:|--------------:|
| val_winter__baseline__lightgbm-v1                                                |         21.553 |        19.429 |            14.143 |         12.331 |                 12.751 |         11.740 |                 16.795 |                14.061 |       -15.702 |
| val_winter__baseline__lightgbm-v2                                                |          0.000 |         0.000 |             0.000 |          0.000 |                  0.000 |          0.000 |                  0.000 |                 0.000 |         0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |         -0.015 |         0.011 |             0.001 |         -0.005 |                 -0.005 |          0.025 |                 -0.010 |                -0.000 |        -0.027 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |         -0.077 |         0.100 |            -0.907 |          0.076 |                  0.194 |          0.225 |                 -0.251 |                -0.173 |        -0.408 |

## Slice: month (% vs reference, negative = better)

| run                                                                              |   2025-12 (n=279) |   2026-01 (n=279) |   2026-02 (n=252) |
|:---------------------------------------------------------------------------------|------------------:|------------------:|------------------:|
| val_winter__baseline__lightgbm-v1                                                |            19.160 |            11.407 |             7.721 |
| val_winter__baseline__lightgbm-v2                                                |             0.000 |             0.000 |             0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |             0.000 |            -0.003 |            -0.011 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |             0.133 |            -0.523 |             0.131 |

## Slice: hour_ending (% vs reference, negative = better)

| run                                                                              |   HE16 |   HE17 |   HE18 |   HE19 |   HE20 |
|:---------------------------------------------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| val_winter__baseline__lightgbm-v1                                                | 16.675 | 17.274 | 13.679 | 10.762 |  7.634 |
| val_winter__baseline__lightgbm-v2                                                |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | -0.003 | -0.008 | -0.007 | -0.002 |  0.001 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | -0.046 | -0.129 | -0.158 | -0.324 | -0.124 |

## Value-focused metrics

`price_weighted_pinball`: pinball loss weighted by the zone's day-ahead price that hour (skill where money is at stake). `p50_vs_ERCOT_%`: mean absolute error of the predicted median vs ERCOT's own forecast (which implies 0 error); negative = the median corrects ERCOT. `skill_vs_same_range_every_day_%`: pinball improvement over the train-period p10/p50/p90 for that zone and hour.

| run                                                                              |   price_weighted_pinball |   price_weighted_vs_ref_% | pw_ci95_%     |   p50_MAE_MW |   ERCOT_MAE_MW |   p50_vs_ERCOT_% |   skill_vs_same_range_every_day_% |
|:---------------------------------------------------------------------------------|-------------------------:|--------------------------:|:--------------|-------------:|---------------:|-----------------:|----------------------------------:|
| val_winter__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |                   206.99 |                     -0.49 | [-1.2, -0.1]  |       339.49 |         468.71 |           -27.57 |                             28.57 |
| val_winter__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |                   208.00 |                     -0.00 | [-0.0, +0.0]  |       339.78 |         468.71 |           -27.51 |                             28.46 |
| val_winter__baseline__lightgbm-v2                                                |                   208.01 |                      0.00 | [+0.0, +0.0]  |       339.79 |         468.71 |           -27.50 |                             28.46 |
| val_winter__baseline__lightgbm-v1                                                |                   219.73 |                      5.64 | [-3.1, +15.5] |       398.51 |         468.71 |           -14.98 |                             19.08 |
