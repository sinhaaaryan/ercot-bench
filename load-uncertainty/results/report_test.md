# Uncertainty eval report: `test`

Reference (declared in advance): **test__baseline__lightgbm-v2**. Negative `vs_ref_%` = lower pinball loss = better. `zone_skill_%` = mean over the 8 weather zones of each zone's % improvement vs the reference (equal weight; positive = better); the system total is reported on its own. CIs: day-clustered bootstrap. Unparseable answers are scored as p10=p50=p90=0 ("trust ERCOT exactly").

## Headline

| run                                                                                 |    n |   days |   pinball_MW |   vs_ref_% | ci95_%         | P(better)   |   zone_skill_% | zone_skill_ci95   |   system_total_skill_% | below_p10   | below_p50   | below_p90   | cov_80   |   width_MW | within_1sd   | within_2sd   | beyond_3sd   |   parse_fail_% |
|:------------------------------------------------------------------------------------|-----:|-------:|-------------:|-----------:|:---------------|:------------|---------------:|:------------------|-----------------------:|:------------|:------------|:------------|:---------|-----------:|:-------------|:-------------|:-------------|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | 1593 |    177 |       127.33 |      -0.00 | [-0.0, +0.0]   | 65.0%       |           0.01 | [-0.0, +0.0]      |                  -0.00 | 6.6%        | 43.7%       | 88.7%       | 82.1%    |    1347.01 | 70.4%        | 95.3%        | 0.8%         |           0.00 |
| test__baseline__lightgbm-v2                                                         | 1593 |    177 |       127.33 |       0.00 | [+0.0, +0.0]   | 0.0%        |           0.00 | [+0.0, +0.0]      |                   0.00 | 6.6%        | 43.6%       | 88.7%       | 82.1%    |    1347.00 | 70.5%        | 95.3%        | 0.8%         |           0.00 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          | 1593 |    177 |       127.33 |       0.00 | [-0.0, +0.0]   | 48.6%       |           0.00 | [-0.0, +0.0]      |                  -0.00 | 6.6%        | 43.7%       | 88.7%       | 82.1%    |    1347.00 | 70.4%        | 95.3%        | 0.8%         |           0.00 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | 1593 |    177 |       127.53 |       0.15 | [-0.1, +0.4]   | 12.8%       |          -0.01 | [-0.2, +0.2]      |                  -0.44 | 6.4%        | 43.0%       | 88.8%       | 82.4%    |    1356.61 | 70.8%        | 95.5%        | 0.8%         |           0.06 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               | 1593 |    177 |       127.57 |       0.19 | [-0.1, +0.5]   | 7.3%        |          -0.12 | [-0.3, +0.1]      |                  -0.34 | 6.4%        | 42.9%       | 88.9%       | 82.5%    |    1357.42 | 71.0%        | 95.5%        | 0.8%         |           0.06 |
| test__baseline__lightgbm-v1                                                         | 1593 |    177 |       133.54 |       4.88 | [+2.8, +7.1]   | 0.0%        |          -7.01 | [-8.5, -5.5]      |                  -3.46 | 7.2%        | 47.8%       | 85.4%       | 78.2%    |    1311.07 | 68.0%        | 93.6%        | 1.0%         |           0.00 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           | 1593 |    177 |       170.56 |      33.95 | [+29.4, +38.8] | 0.0%        |         -97.76 | [-105.2, -91.7]   |                 -13.73 | 6.3%        | 55.2%       | 95.7%       | 89.4%    |    2095.76 | 83.1%        | 96.4%        | 1.1%         |           0.00 |

Calibration targets: below p10 = 10%, below p50 = 50%, below p90 = 90%; within 1 sigma = 68%, within 2 sigma = 95%, beyond 3 sigma = 0.3% (sigma view reads the p10-p90 range as +/-1.28 sigma).

## Calibration by zone (share below p10 / below p90; targets 10% / 90%)

|                                                                                     | coast   | east   | far_west   | north    | north_central   | south   | south_central   | system_total   | west   |
|:------------------------------------------------------------------------------------|:--------|:-------|:-----------|:---------|:----------------|:--------|:----------------|:---------------|:-------|
| test__baseline__lightgbm-v2                                                         | 8%/92%  | 2%/82% | 3%/75%     | 12%/100% | 11%/94%         | 5%/89%  | 7%/91%          | 7%/91%         | 5%/84% |
| test__baseline__lightgbm-v1                                                         | 10%/92% | 4%/74% | 5%/78%     | 12%/99%  | 13%/93%         | 4%/82%  | 8%/87%          | 8%/87%         | 2%/76% |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           | 8%/93%  | 2%/98% | 1%/91%     | 3%/100%  | 14%/97%         | 2%/98%  | 6%/95%          | 20%/91%        | 1%/99% |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               | 7%/93%  | 2%/83% | 3%/76%     | 12%/100% | 10%/94%         | 5%/89%  | 7%/91%          | 6%/90%         | 4%/85% |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          | 8%/92%  | 2%/82% | 3%/75%     | 12%/100% | 11%/94%         | 5%/89%  | 7%/91%          | 7%/91%         | 5%/84% |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      | 7%/93%  | 2%/82% | 3%/76%     | 12%/100% | 10%/94%         | 5%/89%  | 7%/91%          | 6%/90%         | 4%/85% |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged | 8%/92%  | 2%/82% | 3%/75%     | 12%/100% | 11%/94%         | 5%/89%  | 7%/91%          | 7%/91%         | 5%/84% |

## Slice: models_agreed (% vs reference, negative = better)

| run                                                                                 |   False (n=1022) |   True (n=571) |
|:------------------------------------------------------------------------------------|-----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           -0.002 |          0.002 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |            0.088 |          0.277 |
| test__baseline__lightgbm-v1                                                         |            5.199 |          4.262 |
| test__baseline__lightgbm-v2                                                         |            0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |           37.870 |         26.498 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           -0.002 |          0.004 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |            0.227 |          0.119 |

## Slice: big_miss (% vs reference, negative = better)

| run                                                                                 |   False (n=1065) |   True (n=528) |
|:------------------------------------------------------------------------------------|-----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |            0.010 |         -0.013 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |            0.756 |         -0.497 |
| test__baseline__lightgbm-v1                                                         |           -0.945 |         11.155 |
| test__baseline__lightgbm-v2                                                         |            0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |           28.019 |         40.345 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |            0.010 |         -0.011 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |            0.712 |         -0.373 |

## Slice: theme_storms (% vs reference, negative = better)

| run                                                                                 |   False (n=284) |   True (n=1309) |
|:------------------------------------------------------------------------------------|----------------:|----------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.005 |          -0.001 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.428 |           0.224 |
| test__baseline__lightgbm-v1                                                         |           8.892 |           4.390 |
| test__baseline__lightgbm-v2                                                         |           0.000 |           0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |          59.463 |          30.857 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           0.010 |          -0.001 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |          -0.438 |           0.266 |

## Slice: theme_severe (% vs reference, negative = better)

| run                                                                                 |   False (n=978) |   True (n=615) |
|:------------------------------------------------------------------------------------|----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.003 |          0.002 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.084 |          0.415 |
| test__baseline__lightgbm-v1                                                         |           6.527 |          3.056 |
| test__baseline__lightgbm-v2                                                         |           0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |          39.535 |         27.790 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           0.000 |         -0.000 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |          -0.010 |          0.411 |

## Slice: theme_heat (% vs reference, negative = better)

| run                                                                                 |   False (n=964) |   True (n=629) |
|:------------------------------------------------------------------------------------|----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           0.001 |         -0.002 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |           0.158 |          0.147 |
| test__baseline__lightgbm-v1                                                         |           4.270 |          5.609 |
| test__baseline__lightgbm-v2                                                         |           0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |          31.940 |         36.378 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           0.001 |         -0.001 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |           0.239 |          0.131 |

## Slice: theme_heavy_rain (% vs reference, negative = better)

| run                                                                                 |   False (n=1018) |   True (n=575) |
|:------------------------------------------------------------------------------------|-----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           -0.002 |          0.001 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |            0.091 |          0.215 |
| test__baseline__lightgbm-v1                                                         |            6.131 |          3.639 |
| test__baseline__lightgbm-v2                                                         |            0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |           45.073 |         22.975 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           -0.002 |          0.002 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |            0.184 |          0.196 |

## Slice: theme_tropical (% vs reference, negative = better)

| run                                                                                 |   False (n=1354) |   True (n=239) |
|:------------------------------------------------------------------------------------|-----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           -0.001 |          0.002 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |            0.124 |          0.241 |
| test__baseline__lightgbm-v1                                                         |            4.730 |          5.318 |
| test__baseline__lightgbm-v2                                                         |            0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |           37.399 |         23.493 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           -0.002 |          0.008 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |            0.214 |          0.116 |

## Slice: theme_quiet (% vs reference, negative = better)

| run                                                                                 |   False (n=1317) |   True (n=276) |
|:------------------------------------------------------------------------------------|-----------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |           -0.002 |          0.005 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |            0.005 |          0.690 |
| test__baseline__lightgbm-v1                                                         |            4.382 |          6.665 |
| test__baseline__lightgbm-v2                                                         |            0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |           33.929 |         34.021 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |           -0.001 |          0.003 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |            0.051 |          0.694 |

## Slice: zone (% vs reference, negative = better)

| run                                                                                 |   coast (n=177) |   east (n=177) |   far_west (n=177) |   north (n=177) |   north_central (n=177) |   south (n=177) |   south_central (n=177) |   system_total (n=177) |   west (n=177) |
|:------------------------------------------------------------------------------------|----------------:|---------------:|-------------------:|----------------:|------------------------:|----------------:|------------------------:|-----------------------:|---------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |          -0.001 |          0.006 |             -0.008 |          -0.051 |                   0.003 |          -0.011 |                   0.012 |                  0.001 |         -0.005 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |          -0.207 |          0.830 |              0.147 |          -1.261 |                  -0.103 |           0.571 |                   0.371 |                  0.445 |         -0.268 |
| test__baseline__lightgbm-v1                                                         |           7.098 |          5.276 |             -3.734 |          12.922 |                   5.721 |           6.717 |                   4.617 |                  3.462 |         17.449 |
| test__baseline__lightgbm-v2                                                         |           0.000 |          0.000 |              0.000 |           0.000 |                   0.000 |           0.000 |                   0.000 |                  0.000 |          0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |           8.986 |        148.254 |             17.677 |         266.559 |                  20.821 |          70.390 |                  16.572 |                 13.733 |        232.809 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |          -0.003 |          0.003 |             -0.010 |          -0.033 |                   0.003 |           0.003 |                   0.010 |                  0.002 |          0.003 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |          -0.128 |          0.754 |              0.271 |          -1.034 |                   0.064 |           0.763 |                   0.385 |                  0.343 |         -0.100 |

## Slice: month (% vs reference, negative = better)

| run                                                                                 |   2026-03 (n=216) |   2026-04 (n=270) |   2026-05 (n=279) |   2026-06 (n=270) |   2026-07 (n=279) |   2026-08 (n=279) |
|:------------------------------------------------------------------------------------|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |            -0.003 |             0.009 |            -0.004 |            -0.003 |            -0.002 |             0.001 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |            -0.207 |             0.645 |             0.235 |             0.290 |             0.013 |            -0.181 |
| test__baseline__lightgbm-v1                                                         |             9.295 |             1.636 |             5.876 |             4.579 |             5.063 |             3.910 |
| test__baseline__lightgbm-v2                                                         |             0.000 |             0.000 |             0.000 |             0.000 |             0.000 |             0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |            45.945 |            35.115 |            26.118 |            35.397 |            25.299 |            45.621 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |            -0.003 |             0.003 |            -0.002 |             0.000 |            -0.002 |             0.005 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |            -0.103 |             0.384 |             0.457 |             0.274 |             0.256 |            -0.364 |

## Slice: hour_ending (% vs reference, negative = better)

| run                                                                                 |   HE16 |   HE17 |   HE18 |   HE19 |   HE20 |
|:------------------------------------------------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |  0.002 | -0.001 |  0.000 | -0.001 | -0.004 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |  0.104 |  0.163 |  0.101 |  0.074 |  0.342 |
| test__baseline__lightgbm-v1                                                         |  5.537 |  4.948 |  4.347 |  4.875 |  4.693 |
| test__baseline__lightgbm-v2                                                         |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           | 47.200 | 33.624 | 30.333 | 28.949 | 29.655 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |  0.004 |  0.001 |  0.002 | -0.001 | -0.006 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |  0.153 |  0.210 |  0.176 |  0.098 |  0.323 |

## Value-focused metrics

`price_weighted_pinball`: pinball loss weighted by the zone's day-ahead price that hour (skill where money is at stake). `p50_vs_ERCOT_%`: mean absolute error of the predicted median vs ERCOT's own forecast (which implies 0 error); negative = the median corrects ERCOT. `skill_vs_same_range_every_day_%`: pinball improvement over the train-period p10/p50/p90 for that zone and hour.

| run                                                                                 |   price_weighted_pinball |   price_weighted_vs_ref_% | pw_ci95_%      |   p50_MAE_MW |   ERCOT_MAE_MW |   p50_vs_ERCOT_% |   skill_vs_same_range_every_day_% |
|:------------------------------------------------------------------------------------|-------------------------:|--------------------------:|:---------------|-------------:|---------------:|-----------------:|----------------------------------:|
| test-textswap__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged |                   126.96 |                     -0.00 | [-0.0, +0.0]   |       392.69 |         483.65 |           -18.81 |                             17.04 |
| test__openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged          |                   126.97 |                     -0.00 | [-0.0, +0.0]   |       392.69 |         483.65 |           -18.81 |                             17.04 |
| test__baseline__lightgbm-v2                                                         |                   126.97 |                      0.00 | [+0.0, +0.0]   |       392.67 |         483.65 |           -18.81 |                             17.04 |
| test__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged               |                   127.12 |                      0.12 | [-0.2, +0.4]   |       393.57 |         483.65 |           -18.63 |                             16.88 |
| test-textswap__openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged      |                   127.12 |                      0.12 | [-0.1, +0.4]   |       393.59 |         483.65 |           -18.62 |                             16.91 |
| test__baseline__lightgbm-v1                                                         |                   133.16 |                      4.88 | [+2.6, +7.4]   |       411.08 |         483.65 |           -15.00 |                             13.00 |
| test__openai__Qwen_Qwen3-4B-Instruct-2507                                           |                   170.08 |                     33.96 | [+29.5, +39.0] |       469.93 |         483.65 |            -2.84 |                            -11.12 |
