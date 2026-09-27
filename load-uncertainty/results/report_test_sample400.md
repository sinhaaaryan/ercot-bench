# Uncertainty eval report: `test_sample400`

Reference (declared in advance): **test_sample400__baseline__lightgbm-v2**. Negative `vs_ref_%` = lower pinball loss = better. `zone_skill_%` = mean over the 8 weather zones of each zone's % improvement vs the reference (equal weight; positive = better); the system total is reported on its own. CIs: day-clustered bootstrap. Unparseable answers are scored as p10=p50=p90=0 ("trust ERCOT exactly").

## Headline

| run                                                                                  |   n |   days |   pinball_MW |   vs_ref_% | ci95_%         | P(better)   |   zone_skill_% | zone_skill_ci95   |   system_total_skill_% | below_p10   | below_p50   | below_p90   | cov_80   |   width_MW | within_1sd   | within_2sd   | beyond_3sd   |   parse_fail_% |
|:-------------------------------------------------------------------------------------|----:|-------:|-------------:|-----------:|:---------------|:------------|---------------:|:------------------|-----------------------:|:------------|:------------|:------------|:---------|-----------:|:-------------|:-------------|:-------------|---------------:|
| test_sample400-text__claude-cli__claude-opus-5-5                                     | 400 |    162 |       113.70 |      -0.82 | [-2.2, +0.5]   | 88.5%       |          -1.59 | [-3.0, -0.3]      |                   2.00 | 5.3%        | 46.2%       | 89.5%       | 84.2%    |    1296.31 | 74.7%        | 96.7%        | 0.2%         |           0.00 |
| test_sample400__baseline__lightgbm-v2                                                | 400 |    162 |       114.64 |       0.00 | [+0.0, +0.0]   | 0.0%        |           0.00 | [+0.0, +0.0]      |                   0.00 | 5.9%        | 46.8%       | 88.5%       | 82.6%    |    1263.90 | 71.2%        | 95.5%        | 0.5%         |           0.00 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | 400 |    162 |       114.64 |       0.00 | [-0.0, +0.0]   | 38.5%       |           0.00 | [-0.0, +0.0]      |                  -0.00 | 5.8%        | 46.8%       | 88.5%       | 82.7%    |    1263.91 | 71.2%        | 95.5%        | 0.5%         |           0.00 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | 400 |    162 |       114.78 |       0.12 | [-0.4, +0.6]   | 30.3%       |          -0.01 | [-0.5, +0.4]      |                  -0.70 | 5.8%        | 45.9%       | 88.7%       | 82.9%    |    1272.20 | 72.0%        | 95.5%        | 0.8%         |           0.25 |
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  | 400 |    162 |       116.57 |       1.68 | [-0.1, +3.4]   | 2.9%        |          -2.08 | [-3.6, -0.6]      |                  -1.07 | 5.3%        | 44.8%       | 89.5%       | 84.2%    |    1292.08 | 73.6%        | 96.4%        | 0.5%         |           0.00 |
| test_sample400__claude-cli__claude-opus-5-5                                          | 400 |    162 |       116.71 |       1.81 | [+0.0, +3.7]   | 2.3%        |          -2.90 | [-4.7, -1.4]      |                  -1.13 | 4.8%        | 45.2%       | 89.3%       | 84.5%    |    1304.40 | 74.5%        | 96.6%        | 0.2%         |           0.00 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            | 400 |    162 |       117.15 |       2.19 | [+1.0, +3.3]   | 0.0%        |          -4.01 | [-5.3, -2.7]      |                  -0.64 | 5.0%        | 45.9%       | 90.1%       | 85.1%    |    1310.22 | 74.8%        | 96.5%        | 0.8%         |           0.00 |
| test_sample400__baseline__lightgbm-v1                                                | 400 |    162 |       122.17 |       6.57 | [+3.0, +10.2]  | 0.0%        |          -8.37 | [-10.9, -5.5]     |                  -2.58 | 6.4%        | 51.4%       | 85.5%       | 79.1%    |    1251.12 | 68.3%        | 94.8%        | 0.7%         |           0.00 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  | 400 |    162 |       161.84 |      41.17 | [+33.7, +49.6] | 0.0%        |        -102.40 | [-115.9, -90.9]   |                 -12.85 | 5.8%        | 56.9%       | 95.5%       | 89.7%    |    2078.11 | 84.2%        | 96.7%        | 0.8%         |           0.00 |

Calibration targets: below p10 = 10%, below p50 = 50%, below p90 = 90%; within 1 sigma = 68%, within 2 sigma = 95%, beyond 3 sigma = 0.3% (sigma view reads the p10-p90 range as +/-1.28 sigma).

## Calibration by zone (share below p10 / below p90; targets 10% / 90%)

|                                                                                      | coast   | east   | far_west   | north    | north_central   | south   | south_central   | system_total   | west   |
|:-------------------------------------------------------------------------------------|:--------|:-------|:-----------|:---------|:----------------|:--------|:----------------|:---------------|:-------|
| test_sample400__baseline__lightgbm-v2                                                | 9%/93%  | 3%/83% | 3%/74%     | 9%/99%   | 11%/98%         | 1%/88%  | 10%/91%         | 4%/90%         | 4%/82% |
| test_sample400__baseline__lightgbm-v1                                                | 9%/92%  | 3%/75% | 2%/80%     | 11%/100% | 17%/96%         | 1%/81%  | 11%/91%         | 4%/86%         | 1%/72% |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  | 10%/93% | 1%/99% | 0%/90%     | 2%/100%  | 16%/100%        | 2%/95%  | 8%/95%          | 14%/89%        | 0%/98% |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | 9%/93%  | 4%/84% | 3%/75%     | 10%/99%  | 11%/98%         | 1%/88%  | 10%/91%         | 2%/90%         | 4%/82% |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | 9%/93%  | 3%/83% | 3%/74%     | 9%/99%   | 11%/98%         | 1%/88%  | 9%/91%          | 4%/90%         | 4%/82% |
| test_sample400__claude-cli__claude-opus-5-5                                          | 9%/92%  | 3%/85% | 2%/82%     | 5%/100%  | 11%/98%         | 0%/87%  | 10%/91%         | 2%/88%         | 3%/84% |
| test_sample400-text__claude-cli__claude-opus-5-5                                     | 8%/91%  | 2%/84% | 2%/75%     | 4%/100%  | 16%/99%         | 1%/92%  | 10%/94%         | 2%/88%         | 4%/84% |
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  | 8%/92%  | 3%/83% | 1%/83%     | 6%/100%  | 10%/98%         | 3%/86%  | 11%/91%         | 3%/88%         | 3%/87% |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            | 7%/92%  | 2%/89% | 2%/76%     | 5%/100%  | 15%/99%         | 1%/88%  | 8%/89%          | 4%/90%         | 2%/88% |

## Slice: models_agreed (% vs reference, negative = better)

| run                                                                                  |   False (n=272) |   True (n=128) |
|:-------------------------------------------------------------------------------------|----------------:|---------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           1.397 |          2.406 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -0.526 |         -1.551 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           2.376 |          1.718 |
| test_sample400__baseline__lightgbm-v1                                                |           7.444 |          4.368 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           1.699 |          2.094 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          43.906 |         34.304 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.002 |          0.011 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.380 |         -0.526 |

## Slice: big_miss (% vs reference, negative = better)

| run                                                                                  |   False (n=277) |   True (n=123) |
|:-------------------------------------------------------------------------------------|----------------:|---------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           0.972 |          2.508 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -1.188 |         -0.389 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           1.189 |          3.345 |
| test_sample400__baseline__lightgbm-v1                                                |           2.367 |         11.430 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           0.638 |          3.169 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          36.751 |         46.292 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.016 |         -0.016 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.484 |         -0.296 |

## Slice: theme_storms (% vs reference, negative = better)

| run                                                                                  |   False (n=80) |   True (n=320) |
|:-------------------------------------------------------------------------------------|---------------:|---------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |         -2.813 |          2.313 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          0.001 |         -0.932 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |          0.464 |          2.430 |
| test_sample400__baseline__lightgbm-v1                                                |          8.158 |          6.347 |
| test_sample400__baseline__lightgbm-v2                                                |          0.000 |          0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |         -3.248 |          2.518 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |         67.062 |         37.556 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          0.010 |          0.000 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |         -0.556 |          0.217 |

## Slice: theme_severe (% vs reference, negative = better)

| run                                                                                  |   False (n=265) |   True (n=135) |
|:-------------------------------------------------------------------------------------|----------------:|---------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           2.461 |          0.111 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -1.528 |          0.620 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           2.388 |          1.785 |
| test_sample400__baseline__lightgbm-v1                                                |           5.893 |          7.939 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           2.023 |          1.383 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          42.640 |         38.207 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.001 |          0.006 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.014 |          0.342 |

## Slice: theme_heat (% vs reference, negative = better)

| run                                                                                  |   False (n=248) |   True (n=152) |
|:-------------------------------------------------------------------------------------|----------------:|---------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           0.639 |          2.861 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -0.397 |         -1.291 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           2.808 |          1.492 |
| test_sample400__baseline__lightgbm-v1                                                |           5.137 |          8.182 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           2.082 |          1.507 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          41.612 |         40.681 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.003 |         -0.000 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.030 |          0.226 |

## Slice: theme_heavy_rain (% vs reference, negative = better)

| run                                                                                  |   False (n=246) |   True (n=154) |
|:-------------------------------------------------------------------------------------|----------------:|---------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           1.051 |          2.327 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -0.036 |         -1.612 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           3.476 |          0.881 |
| test_sample400__baseline__lightgbm-v1                                                |           6.225 |          6.918 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |          0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           0.155 |          3.494 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          54.048 |         28.097 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.000 |          0.002 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.256 |          0.507 |

## Slice: theme_tropical (% vs reference, negative = better)

| run                                                                                  |   False (n=336) |   True (n=64) |
|:-------------------------------------------------------------------------------------|----------------:|--------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           1.391 |         2.463 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -0.811 |        -0.836 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           2.558 |         1.207 |
| test_sample400__baseline__lightgbm-v1                                                |           8.033 |         2.686 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |         0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           1.458 |         2.748 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          46.926 |        25.911 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.000 |         0.004 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.073 |         0.253 |

## Slice: theme_quiet (% vs reference, negative = better)

| run                                                                                  |   False (n=337) |   True (n=63) |
|:-------------------------------------------------------------------------------------|----------------:|--------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           2.047 |         0.256 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |          -0.375 |        -2.556 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |           2.132 |         2.411 |
| test_sample400__baseline__lightgbm-v1                                                |           6.092 |         8.445 |
| test_sample400__baseline__lightgbm-v2                                                |           0.000 |         0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           2.259 |         0.050 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |          40.695 |        43.058 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.001 |         0.003 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.049 |         0.410 |

## Slice: zone (% vs reference, negative = better)

| run                                                                                  |   coast (n=50) |   east (n=43) |   far_west (n=47) |   north (n=44) |   north_central (n=43) |   south (n=45) |   south_central (n=39) |   system_total (n=40) |   west (n=49) |
|:-------------------------------------------------------------------------------------|---------------:|--------------:|------------------:|---------------:|-----------------------:|---------------:|-----------------------:|----------------------:|--------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |          2.520 |         2.018 |            -2.473 |         -3.480 |                  1.328 |          6.861 |                  4.438 |                 1.066 |         5.429 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |         -2.728 |         3.976 |             0.793 |          7.976 |                 -0.847 |          0.288 |                 -1.425 |                -1.997 |         4.694 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |          1.480 |         6.709 |            -0.291 |          9.326 |                  3.385 |          7.573 |                  1.940 |                 0.645 |         1.981 |
| test_sample400__baseline__lightgbm-v1                                                |          6.446 |         1.077 |            -4.454 |         12.881 |                 17.797 |         11.668 |                  1.485 |                 2.582 |        20.086 |
| test_sample400__baseline__lightgbm-v2                                                |          0.000 |         0.000 |             0.000 |          0.000 |                  0.000 |          0.000 |                  0.000 |                 0.000 |         0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |          2.428 |         2.184 |            -2.627 |         -1.547 |                 -0.328 |          9.897 |                  6.197 |                 1.134 |         7.013 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |         10.162 |       160.119 |            17.523 |        249.238 |                 41.737 |         90.831 |                 19.076 |                12.846 |       230.541 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |         -0.004 |        -0.047 |            -0.018 |         -0.004 |                  0.017 |          0.032 |                 -0.002 |                 0.002 |         0.020 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |         -0.439 |         0.938 |            -0.038 |         -0.559 |                 -0.008 |          0.987 |                 -0.207 |                 0.696 |        -0.611 |

## Slice: month (% vs reference, negative = better)

| run                                                                                  |   2026-03 (n=60) |   2026-04 (n=70) |   2026-05 (n=66) |   2026-06 (n=69) |   2026-07 (n=70) |   2026-08 (n=65) |
|:-------------------------------------------------------------------------------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |           -1.018 |           -1.575 |            0.072 |            2.885 |            2.040 |            5.075 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     |            1.261 |           -0.699 |            0.397 |           -3.013 |            1.068 |           -1.582 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |            1.575 |            3.339 |            4.590 |            0.896 |           -0.202 |            3.631 |
| test_sample400__baseline__lightgbm-v1                                                |           10.702 |            6.543 |            6.828 |            4.378 |           10.154 |            4.028 |
| test_sample400__baseline__lightgbm-v2                                                |            0.000 |            0.000 |            0.000 |            0.000 |            0.000 |            0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |           -0.954 |            0.558 |            1.230 |            2.838 |            3.351 |            2.142 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |           53.333 |           32.069 |           55.240 |           29.800 |           40.387 |           48.373 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           -0.011 |            0.015 |            0.001 |            0.003 |           -0.006 |            0.000 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           -0.571 |            0.432 |            0.271 |            0.494 |            0.312 |           -0.489 |

## Slice: hour_ending (% vs reference, negative = better)

| run                                                                                  |   HE16 |   HE17 |   HE18 |   HE19 |   HE20 |
|:-------------------------------------------------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |  0.827 |  1.631 |  1.277 |  2.050 |  2.747 |
| test_sample400-text__claude-cli__claude-opus-5-5                                     | -0.301 | -1.012 | -1.262 | -0.639 | -0.830 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |  2.511 |  2.076 |  1.901 |  2.086 |  2.412 |
| test_sample400__baseline__lightgbm-v1                                                |  6.520 |  7.241 |  5.401 |  6.815 |  6.946 |
| test_sample400__baseline__lightgbm-v2                                                |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |
| test_sample400__claude-cli__claude-opus-5-5                                          |  1.348 |  1.388 |  1.162 |  2.448 |  2.861 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  | 55.386 | 40.320 | 37.530 | 36.292 | 36.286 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |  0.000 |  0.006 |  0.005 |  0.004 | -0.009 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |  0.146 |  0.282 |  0.045 | -0.113 |  0.258 |

## Value-focused metrics

`price_weighted_pinball`: pinball loss weighted by the zone's day-ahead price that hour (skill where money is at stake). `p50_vs_ERCOT_%`: mean absolute error of the predicted median vs ERCOT's own forecast (which implies 0 error); negative = the median corrects ERCOT. `skill_vs_same_range_every_day_%`: pinball improvement over the train-period p10/p50/p90 for that zone and hour.

| run                                                                                  |   price_weighted_pinball |   price_weighted_vs_ref_% | pw_ci95_%      |   p50_MAE_MW |   ERCOT_MAE_MW |   p50_vs_ERCOT_% |   skill_vs_same_range_every_day_% |
|:-------------------------------------------------------------------------------------|-------------------------:|--------------------------:|:---------------|-------------:|---------------:|-----------------:|----------------------------------:|
| test_sample400-text__claude-cli__claude-opus-5-5                                     |                   120.44 |                     -0.35 | [-1.8, +0.9]   |       355.13 |         454.13 |           -21.80 |                             21.64 |
| test_sample400__baseline__lightgbm-v2                                                |                   120.86 |                      0.00 | [+0.0, +0.0]   |       355.57 |         454.13 |           -21.70 |                             20.99 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |                   120.86 |                      0.00 | [-0.0, +0.0]   |       355.61 |         454.13 |           -21.69 |                             20.99 |
| test_sample400__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |                   121.01 |                      0.12 | [-0.3, +0.6]   |       355.07 |         454.13 |           -21.81 |                             20.90 |
| test_sample400-numeric__claude-cli__claude-opus-5-5                                  |                   123.28 |                      2.00 | [+0.3, +3.7]   |       359.25 |         454.13 |           -20.89 |                             19.66 |
| test_sample400__claude-cli__claude-opus-5-5                                          |                   123.43 |                      2.13 | [+0.3, +4.2]   |       361.01 |         454.13 |           -20.50 |                             19.56 |
| test_sample400-textswap_text__claude-cli__claude-opus-5-5                            |                   123.79 |                      2.43 | [+1.2, +3.7]   |       358.07 |         454.13 |           -21.15 |                             19.27 |
| test_sample400__baseline__lightgbm-v1                                                |                   129.45 |                      7.11 | [+3.1, +11.1]  |       381.16 |         454.13 |           -16.07 |                             15.80 |
| test_sample400__openai__Qwen_Qwen3-4B-Instruct-2507                                  |                   168.86 |                     39.72 | [+31.2, +48.7] |       445.28 |         454.13 |            -1.95 |                            -11.54 |
