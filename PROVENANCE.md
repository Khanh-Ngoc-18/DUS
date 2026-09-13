# Provenance — paper.md

Moi so trong bai doi chieu voi file ket qua hoac tinh lai tu log tho. **320 / 491 khop.**

Sinh boi `python audit_paper.py`. Script chi doc, khong tinh lai bat ky phep tinh nao cua pipeline.

| Section | Claim | Y nghia | Nguon | Gia tri nguon | Khop |
| :---- | ----: | :---- | :---- | ----: | :----: |
| IV-B | `4500` | so debate | `verify_threshold.json :: sequential.n_rollouts` | 4500 | OK |
| IV-B | `27000` | tong so round | `verify_threshold.json :: naive.n_rounds` | 27000 | OK |
| IV-B | `22500` | so round khong-cuoi | `verify_threshold.json :: nonfinal.n` | 22500 | OK |
| IV-B | `32.9` | ty le sai tren round khong-cuoi | `verify_threshold.json :: nonfinal.error_rate` | 0.329333 | OK |
| IV-B | `0.326` | base rate cua nhan da sua | `p0_critic_eval.json :: base_rate` | 0.325733 | OK |
| IV-C | `18978` | so round trong split train | `dus_weights.json :: n_rows` | 18978 | OK |
| IV-C | `2748` | so cau hoi doc lap | `p1_power.json :: n_samples` | 2748 | OK |
| IV-C | `0.958` | power tai 65 cau | `p1_power.json :: power.65.power` | 0.958333 | OK |
| IV-C | `0.992` | power tai 100 cau | `p1_power.json :: power.100.power` | 0.991667 | OK |
| IV-C | `1.000` | power tai 200 cau | `p1_power.json :: power.200.power` | 1 | OK |
| V-B | `33.9` | ty le round bi gan 'stop' (giao thuc truc tiep) | `verify_threshold.json :: naive.stop_rate` | 0.338852 | OK |
| V-B | `9149` | so round duoi nguong | `verify_threshold.json :: circularity.n_below_threshold` | 9149 | OK |
| V-B | `-0.604` | nguong T* chon boi giao thuc truc tiep | `verify_threshold.json :: naive.threshold` | -0.604351 | OK |
| V-B | `4.85` | error rate theo ROUND | `verify_threshold.json :: sequential.miss_rate_per_row` | 0.0484815 | OK |
| V-B | `5.09` | error rate theo DEBATE | `verify_threshold.json :: sequential.miss_rate_per_sample` | 0.0508889 | OK |
| V-B | `1.05` | he so thoi phong per-round -> per-debate | `verify_threshold.json :: sequential.inflation` | 1.04966 | OK |
| V-B | `1517` | so debate co flag fire | `verify_threshold.json :: sequential.n_flag_fired` | 1517 | OK |
| V-B | `14987` | so round he chuan (consensus) thuc su chay | `verify_threshold.json :: sequential.rounds_standard_system` | 14987 | OK |
| V-B | `0` | so debate dung SOM HON he chuan | `verify_threshold.json :: sequential.n_stopped_earlier` | 0 | OK |
| V-B | `0.0` | ty le round tiet kiem THUC | `verify_threshold.json :: sequential.real_saving_rate` | 0 | OK |
| V-B | `11754` | so round co answer_entropy = 0 (= consensus) | `verify_threshold.json :: circularity.n_entropy_zero` | 11754 | OK |
| V-B | `2,037` | round entropy=0 la round CUOI cua debate | `verify_threshold.json :: circularity.n_entropy_zero_is_last` | 2037 | OK |
| V-B | `872` | so round nam dung tai T (mass point) | `verify_threshold.json :: mass_point.n_rows_equal_T` | 872 | OK |
| V-B | `37.1` | stop rate neu doi dau < thanh <= | `verify_threshold.json :: mass_point.stop_rate_le` | 0.371148 | OK |
| V-B | `5.63` | error rate neu doi dau < thanh <= | `verify_threshold.json :: mass_point.miss_rate_le` | 0.0562963 | OK |
| V-B | `0.717` | AUC tren MOI round (thoi phong) | `verify_threshold.json :: nonfinal.auc_all_rounds` | 0.716607 | OK |
| V-B | `0.714` | AUC tren round khong-cuoi | `verify_threshold.json :: nonfinal.auc_nonfinal` | 0.713549 | OK |
| V-B | `0.003` | do thoi phong AUC | `verify_threshold.json :: nonfinal.auc_inflation` | 0.00305798 | OK |
| V-B | `0.323` | uncertain rate tren train | `dus_weights.json :: uncertain_rate` | 0.323111 | OK |
| V-C | `0.705` | ROC-AUC model 'base (DUS 4 feature)' | `p0_critic_eval.json :: models['base (DUS 4 feature)'].auc` | 0.704715 | OK |
| V-C | `0.729` | ROC-AUC model 'critic only' | `p0_critic_eval.json :: models['critic only'].auc` | 0.729376 | OK |
| V-C | `0.736` | ROC-AUC model 'base + critic' | `p0_critic_eval.json :: models['base + critic'].auc` | 0.736075 | OK |
| V-C | `0.738` | ROC-AUC DUS-11 5-fold CV theo cau | `p1_power.json :: auc_cv` | 0.737609 | OK |
| V-C | `0.882` | AUC per-task gsm8k | `p0_critic_eval.json :: per_task.gsm8k` | 0.882276 | OK |
| V-C | `0.705` | AUC per-task mmlu | `p0_critic_eval.json :: per_task.mmlu` | 0.704874 | OK |
| V-C | `0.590` | AUC per-task strategyqa | `p0_critic_eval.json :: per_task.strategyqa` | 0.589659 | OK |
| V-C | `0.025` | hieu AUC ghep cap, CRITIC-7 - DUS-4 | `p0_critic_eval.json :: paired['critic only - base (DUS 4 feature)'].delta` | 0.0246616 | OK |
| V-C | `0.007` | CI duoi cua hieu, CRITIC-7 - DUS-4 | `p0_critic_eval.json :: paired['critic only - base (DUS 4 feature)'].ci[0]` | 0.00730541 | OK |
| V-C | `0.042` | CI tren cua hieu, CRITIC-7 - DUS-4 | `p0_critic_eval.json :: paired['critic only - base (DUS 4 feature)'].ci[1]` | 0.0424755 | OK |
| V-C | `0.002` | p hai phia, CRITIC-7 - DUS-4 | `p0_critic_eval.json :: paired['critic only - base (DUS 4 feature)'].p_two_sided` | 0.002 | OK |
| V-C | `0.031` | hieu AUC ghep cap, DUS-11 - DUS-4 | `p0_critic_eval.json :: paired['base + critic - base (DUS 4 feature)'].delta` | 0.0313602 | OK |
| V-C | `0.021` | CI duoi cua hieu, DUS-11 - DUS-4 | `p0_critic_eval.json :: paired['base + critic - base (DUS 4 feature)'].ci[0]` | 0.0205175 | OK |
| V-C | `0.043` | CI tren cua hieu, DUS-11 - DUS-4 | `p0_critic_eval.json :: paired['base + critic - base (DUS 4 feature)'].ci[1]` | 0.0430659 | OK |
| V-C | `0.000` | p hai phia, DUS-11 - DUS-4 | `p0_critic_eval.json :: paired['base + critic - base (DUS 4 feature)'].p_two_sided` | 0 | OK |
| V-C | `0.007` | hieu AUC ghep cap, DUS-11 - CRITIC-7 | `p0_critic_eval.json :: paired['base + critic - critic only'].delta` | 0.00669854 | OK |
| V-C | `-0.004` | CI duoi cua hieu, DUS-11 - CRITIC-7 | `p0_critic_eval.json :: paired['base + critic - critic only'].ci[0]` | -0.00360753 | OK |
| V-C | `0.017` | CI tren cua hieu, DUS-11 - CRITIC-7 | `p0_critic_eval.json :: paired['base + critic - critic only'].ci[1]` | 0.0173497 | OK |
| V-C | `0.202` | p hai phia, DUS-11 - CRITIC-7 | `p0_critic_eval.json :: paired['base + critic - critic only'].p_two_sided` | 0.202 | OK |
| V-D | `56.6` | % debate mmlu q50 dung SOM HON consensus | `p3_holdout_policy.json :: nested_cv.mmlu.stop_timing['unc<q50'].earlier_pct` | 57.8667 | **LECH** |
| V-E | `-39.4` | dToken mmlu unc<q50 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.mmlu.paired_vs_consensus['unc<q50'].dtok.mean` | -41.0658 | **LECH** |
| V-E | `-0.001` | dAccuracy mmlu unc<q50 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.mmlu.paired_vs_consensus['unc<q50'].dacc.mean` | 0.000666667 | **LECH** |
| V-E | `-26.1` | dToken mmlu unc<q40 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.mmlu.paired_vs_consensus['unc<q40'].dtok.mean` | -28.2215 | **LECH** |
| V-E | `-0.003` | dAccuracy mmlu unc<q40 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.mmlu.paired_vs_consensus['unc<q40'].dacc.mean` | -0.00133333 | **LECH** |
| V-E | `-6.0` | dToken gsm8k unc<q50 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.gsm8k.paired_vs_consensus['unc<q50'].dtok.mean` | -6.84366 | **LECH** |
| V-E | `-0.001` | dAccuracy gsm8k unc<q50 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.gsm8k.paired_vs_consensus['unc<q50'].dacc.mean` | 0.000666667 | **LECH** |
| V-E | `-0.4` | dToken strategyqa unc<q50 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.strategyqa.paired_vs_consensus['unc<q50'].dtok.mean` | -1.88818 | **LECH** |
| V-E | `-0.001` | dAccuracy strategyqa unc<q50 vs consensus (nested CV) | `p3_holdout_policy.json :: nested_cv.strategyqa.paired_vs_consensus['unc<q50'].dacc.mean` | 0.00333333 | **LECH** |
| V-F | `0.51` | TABLE IX net tong theo % | `p4_rescue_hurt.json :: total.net_pct` | 0.511111 | OK |
| V-F | `116` | so cap khong khop cua gsm8k (rescued + corrupted) | `p4_rescue_hurt.json :: gsm8k.n_discordant` | 116 | OK |
| V-E | `2.3` | SD cua dToken mmlu q50 qua 5 seed | `p3_holdout_policy.json :: nested_cv.mmlu.paired_vs_consensus['unc<q50'].dtok.sd` | 2.19059 | **LECH** |
| V-E | `-39.2` | dToken mmlu q50 tren holdout 70/20/10 | `p3_holdout_policy.json :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dtok.mean` | -41.6626 | **LECH** |
| V-E | `14.6` | SD cua dToken mmlu q50 tren holdout | `p3_holdout_policy.json :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dtok.sd` | 12.7408 | **LECH** |
| V-E | `-0.046` | dAccuracy mmlu q50 tren holdout 70/20/10 | `p3_holdout_policy.json :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dacc.mean` | -0.0164465 | **LECH** |
| V-E | `-0.096` | CI cua dAcc mmlu q50 tren holdout, can duoi | `p3_holdout_policy.json :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dacc_ci_clustered[0]` | -0.0748427 | **LECH** |
| V-E | `-0.007` | CI cua dAcc mmlu q50 tren holdout, can tren | `p3_holdout_policy.json :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dacc_ci_clustered[1]` | 0.0314515 | **LECH** |
| V-E | `151` | so debate test cua mmlu | `p3_holdout_policy.json :: fixed_holdout.mmlu.n_debate` | 151 | OK |
| V-E | `0.587` | accuracy cua `always` tren tap test MMLU | `p3_holdout_policy.json :: fixed_holdout.mmlu.policies.always.acc.mean` | 0.586789 | OK |
| V-E | `-40.9` | dToken mmlu q50 neu chon T in-sample | `p3_holdout_policy.json :: threshold_selection_optimism.mmlu['unc<q50'].in_sample_dtok` | -43.4479 | **LECH** |
| V-E | `3.2` | do lac quan LON NHAT cua dToken khi chon T in-sample | `p3_holdout_policy.json :: max |threshold_selection_optimism.*.*.dtok_pp|` | 3.7321 | **LECH** |
| V-E | `0.004` | do lac quan LON NHAT cua dAccuracy khi chon T in-sample | `p3_holdout_policy.json :: max |threshold_selection_optimism.*.*.dacc|` | 0.006 | **LECH** |
| V-F | `0.516` | McNemar p, net cua gsm8k khac 0 | `p4_rescue_hurt.json :: gsm8k.mcnemar_p` | 0.515913 | OK |
| V-F | `0.074` | McNemar p, net cua mmlu khac 0 | `p4_rescue_hurt.json :: mmlu.mcnemar_p` | 0.0738758 | OK |
| V-F | `0.764` | McNemar p, net cua strategyqa khac 0 | `p4_rescue_hurt.json :: strategyqa.mcnemar_p` | 0.763774 | OK |
| V-F | `0.320` | McNemar p, net cua total khac 0 | `p4_rescue_hurt.json :: total.mcnemar_p` | 0.319798 | OK |
| VI-A | `0.875` | AUC gop qua seed, gsm8k | `p2_cost_accuracy.json :: pooled_auc_across_seeds.gsm8k.auc` | 0.874917 | OK |
| VI-A | `56.0` | ty le token cua consensus, gsm8k | `p3_holdout_policy.json :: nested_cv.gsm8k.policies.consensus.tok.mean` | 55.9805 | OK |
| VI-A | `0.685` | AUC gop qua seed, mmlu | `p2_cost_accuracy.json :: pooled_auc_across_seeds.mmlu.auc` | 0.68509 | OK |
| VI-A | `68.2` | ty le token cua consensus, mmlu | `p3_holdout_policy.json :: nested_cv.mmlu.policies.consensus.tok.mean` | 68.21 | OK |
| VI-A | `0.609` | AUC gop qua seed, strategyqa | `p2_cost_accuracy.json :: pooled_auc_across_seeds.strategyqa.auc` | 0.608987 | OK |
| VI-A | `48.5` | ty le token cua consensus, strategyqa | `p3_holdout_policy.json :: nested_cv.strategyqa.policies.consensus.tok.mean` | 48.4502 | OK |
| TABLE II | `0.638` | TABLE II he so tho answer_entropy | `(o bang)` | 0.637886 | OK |
| TABLE II | `0.638` | TABLE II trong so cuoi answer_entropy | `(o bang)` | 0.638498 | OK |
| TABLE II | `0.194` | TABLE II he so tho answer_flip_rate | `(o bang)` | 0.193757 | OK |
| TABLE II | `0.194` | TABLE II trong so cuoi answer_flip_rate | `(o bang)` | 0.193943 | OK |
| TABLE II | `0.167` | TABLE II he so tho disagreement_persistence | `(o bang)` | 0.167399 | OK |
| TABLE II | `0.168` | TABLE II trong so cuoi disagreement_persistence | `(o bang)` | 0.16756 | OK |
| TABLE II | `-0.043` | TABLE II he so tho confidence_variance | `(o bang)` | -0.0426779 | OK |
| TABLE II | `0.000` | TABLE II trong so cuoi confidence_variance | `(o bang)` | 0 | OK |
| TABLE IV | `0.694` | TABLE IV AUC answer_entropy | `(o bang)` | 0.694038 | OK |
| TABLE IV | `0.194` | TABLE IV |AUC-0.5| answer_entropy | `(o bang)` | 0.194038 | OK |
| TABLE IV | `0.309` | TABLE IV AUC verdict_conf_mean | `(o bang)` | 0.30868 | OK |
| TABLE IV | `0.191` | TABLE IV |AUC-0.5| verdict_conf_mean | `(o bang)` | 0.19132 | OK |
| TABLE IV | `0.690` | TABLE IV AUC n_disagree | `(o bang)` | 0.69004 | OK |
| TABLE IV | `0.190` | TABLE IV |AUC-0.5| n_disagree | `(o bang)` | 0.19004 | OK |
| TABLE IV | `0.664` | TABLE IV AUC disagreement_persistence | `(o bang)` | 0.664104 | OK |
| TABLE IV | `0.164` | TABLE IV |AUC-0.5| disagreement_persistence | `(o bang)` | 0.164104 | OK |
| TABLE IV | `0.349` | TABLE IV AUC verdict_conf_min | `(o bang)` | 0.348901 | OK |
| TABLE IV | `0.151` | TABLE IV |AUC-0.5| verdict_conf_min | `(o bang)` | 0.151099 | OK |
| TABLE IV | `0.624` | TABLE IV AUC critic_vs_majority | `(o bang)` | 0.624187 | OK |
| TABLE IV | `0.124` | TABLE IV |AUC-0.5| critic_vs_majority | `(o bang)` | 0.124187 | OK |
| TABLE IV | `0.597` | TABLE IV AUC answer_flip_rate | `(o bang)` | 0.597184 | OK |
| TABLE IV | `0.097` | TABLE IV |AUC-0.5| answer_flip_rate | `(o bang)` | 0.0971838 | OK |
| TABLE IV | `0.446` | TABLE IV AUC critic_conf | `(o bang)` | 0.446211 | OK |
| TABLE IV | `0.054` | TABLE IV |AUC-0.5| critic_conf | `(o bang)` | 0.0537892 | OK |
| TABLE IV | `0.533` | TABLE IV AUC confidence_variance | `(o bang)` | 0.533155 | OK |
| TABLE IV | `0.033` | TABLE IV |AUC-0.5| confidence_variance | `(o bang)` | 0.0331547 | OK |
| TABLE IV | `0.522` | TABLE IV AUC conf_gap_critic_solvers | `(o bang)` | 0.521797 | OK |
| TABLE IV | `0.022` | TABLE IV |AUC-0.5| conf_gap_critic_solvers | `(o bang)` | 0.0217969 | OK |
| TABLE IV | `0.517` | TABLE IV AUC critic_alone | `(o bang)` | 0.517364 | OK |
| TABLE IV | `0.017` | TABLE IV |AUC-0.5| critic_alone | `(o bang)` | 0.0173643 | OK |
| TABLE V | `0.705` | TABLE V AUC DUS-4 — consensus dynamics | `(o bang)` | 0.704715 | OK |
| TABLE V | `0.671` | TABLE V CI duoi DUS-4 — consensus dynamics | `(o bang)` | 0.671496 | OK |
| TABLE V | `0.737` | TABLE V CI tren DUS-4 — consensus dynamics | `(o bang)` | 0.736858 | OK |
| TABLE V | `0.729` | TABLE V AUC CRITIC-7 — critic signals alone | `(o bang)` | 0.729376 | OK |
| TABLE V | `0.703` | TABLE V CI duoi CRITIC-7 — critic signals alone | `(o bang)` | 0.702592 | OK |
| TABLE V | `0.757` | TABLE V CI tren CRITIC-7 — critic signals alone | `(o bang)` | 0.756652 | OK |
| TABLE V | `0.736` | TABLE V AUC DUS-11 — both families | `(o bang)` | 0.736075 | OK |
| TABLE V | `0.707` | TABLE V CI duoi DUS-11 — both families | `(o bang)` | 0.706771 | OK |
| TABLE V | `0.764` | TABLE V CI tren DUS-11 — both families | `(o bang)` | 0.764468 | OK |
| TABLE V | `0.738` | TABLE V AUC 5-fold CV | `(o bang)` | 0.737609 | OK |
| TABLE V | `0.720` | TABLE V CI duoi 5-fold CV | `(o bang)` | 0.720477 | OK |
| TABLE V | `0.754` | TABLE V CI tren 5-fold CV | `(o bang)` | 0.753704 | OK |
| TABLE Va | `0.025` | TABLE Va delta CRITIC-7 - DUS-4 | `(o bang)` | 0.0246616 | OK |
| TABLE Va | `0.007` | TABLE Va CI duoi CRITIC-7 - DUS-4 | `(o bang)` | 0.00730541 | OK |
| TABLE Va | `0.042` | TABLE Va CI tren CRITIC-7 - DUS-4 | `(o bang)` | 0.0424755 | OK |
| TABLE Va | `0.031` | TABLE Va delta DUS-11 - DUS-4 | `(o bang)` | 0.0313602 | OK |
| TABLE Va | `0.021` | TABLE Va CI duoi DUS-11 - DUS-4 | `(o bang)` | 0.0205175 | OK |
| TABLE Va | `0.043` | TABLE Va CI tren DUS-11 - DUS-4 | `(o bang)` | 0.0430659 | OK |
| TABLE Va | `0.007` | TABLE Va delta DUS-11 - CRITIC-7 | `(o bang)` | 0.00669854 | OK |
| TABLE Va | `-0.004` | TABLE Va CI duoi DUS-11 - CRITIC-7 | `(o bang)` | -0.00360753 | OK |
| TABLE Va | `0.017` | TABLE Va CI tren DUS-11 - CRITIC-7 | `(o bang)` | 0.0173497 | OK |
| TABLE VI | `200` | TABLE VI gsm8k q0.5 earlier (dem) | `(o bang)` | 208 | **LECH** |
| TABLE VI | `13.3` | TABLE VI gsm8k q0.5 earlier (%) | `(o bang)` | 13.8667 | **LECH** |
| TABLE VI | `1155` | TABLE VI gsm8k q0.5 same (dem) | `(o bang)` | 1147 | **LECH** |
| TABLE VI | `77.0` | TABLE VI gsm8k q0.5 same (%) | `(o bang)` | 76.4667 | **LECH** |
| TABLE VI | `145` | TABLE VI gsm8k q0.5 later (dem) | `(o bang)` | 145 | OK |
| TABLE VI | `9.7` | TABLE VI gsm8k q0.5 later (%) | `(o bang)` | 9.66667 | OK |
| TABLE VI | `149` | TABLE VI strategyqa q0.5 earlier (dem) | `(o bang)` | 159 | **LECH** |
| TABLE VI | `9.9` | TABLE VI strategyqa q0.5 earlier (%) | `(o bang)` | 10.6 | **LECH** |
| TABLE VI | `1133` | TABLE VI strategyqa q0.5 same (dem) | `(o bang)` | 1133 | OK |
| TABLE VI | `75.5` | TABLE VI strategyqa q0.5 same (%) | `(o bang)` | 75.5333 | OK |
| TABLE VI | `218` | TABLE VI strategyqa q0.5 later (dem) | `(o bang)` | 208 | **LECH** |
| TABLE VI | `14.5` | TABLE VI strategyqa q0.5 later (%) | `(o bang)` | 13.8667 | **LECH** |
| TABLE VI | `614` | TABLE VI mmlu q0.4 earlier (dem) | `(o bang)` | 645 | **LECH** |
| TABLE VI | `40.9` | TABLE VI mmlu q0.4 earlier (%) | `(o bang)` | 43 | **LECH** |
| TABLE VI | `809` | TABLE VI mmlu q0.4 same (dem) | `(o bang)` | 783 | **LECH** |
| TABLE VI | `53.9` | TABLE VI mmlu q0.4 same (%) | `(o bang)` | 52.2 | **LECH** |
| TABLE VI | `77` | TABLE VI mmlu q0.4 later (dem) | `(o bang)` | 72 | **LECH** |
| TABLE VI | `5.1` | TABLE VI mmlu q0.4 later (%) | `(o bang)` | 4.8 | **LECH** |
| TABLE VI | `849` | TABLE VI mmlu q0.5 earlier (dem) | `(o bang)` | 868 | **LECH** |
| TABLE VI | `56.6` | TABLE VI mmlu q0.5 earlier (%) | `(o bang)` | 57.8667 | **LECH** |
| TABLE VI | `631` | TABLE VI mmlu q0.5 same (dem) | `(o bang)` | 612 | **LECH** |
| TABLE VI | `42.1` | TABLE VI mmlu q0.5 same (%) | `(o bang)` | 40.8 | **LECH** |
| TABLE VI | `20` | TABLE VI mmlu q0.5 later (dem) | `(o bang)` | 20 | OK |
| TABLE VI | `1.3` | TABLE VI mmlu q0.5 later (%) | `(o bang)` | 1.33333 | OK |
| TABLE VII | `0.779` | TABLE VII accuracy gsm8k always | `(o bang)` | 0.778667 | OK |
| TABLE VII | `100.0` | TABLE VII token% gsm8k always | `(o bang)` | 100 | OK |
| TABLE VII | `0.547` | TABLE VII accuracy mmlu always | `(o bang)` | 0.547333 | OK |
| TABLE VII | `100.0` | TABLE VII token% mmlu always | `(o bang)` | 100 | OK |
| TABLE VII | `0.686` | TABLE VII accuracy strategyqa always | `(o bang)` | 0.686 | OK |
| TABLE VII | `100.0` | TABLE VII token% strategyqa always | `(o bang)` | 100 | OK |
| TABLE VII | `0.785` | TABLE VII accuracy gsm8k consensus | `(o bang)` | 0.785333 | OK |
| TABLE VII | `56.0` | TABLE VII token% gsm8k consensus | `(o bang)` | 55.9805 | OK |
| TABLE VII | `0.546` | TABLE VII accuracy mmlu consensus | `(o bang)` | 0.546 | OK |
| TABLE VII | `68.2` | TABLE VII token% mmlu consensus | `(o bang)` | 68.21 | OK |
| TABLE VII | `0.691` | TABLE VII accuracy strategyqa consensus | `(o bang)` | 0.690667 | OK |
| TABLE VII | `48.5` | TABLE VII token% strategyqa consensus | `(o bang)` | 48.4502 | OK |
| TABLE VII | `0.784` | TABLE VII accuracy gsm8k fixed_k1 | `(o bang)` | 0.784 | OK |
| TABLE VII | `16.7` | TABLE VII token% gsm8k fixed_k1 | `(o bang)` | 16.6667 | OK |
| TABLE VII | `0.530` | TABLE VII accuracy mmlu fixed_k1 | `(o bang)` | 0.53 | OK |
| TABLE VII | `16.7` | TABLE VII token% mmlu fixed_k1 | `(o bang)` | 16.6667 | OK |
| TABLE VII | `0.683` | TABLE VII accuracy strategyqa fixed_k1 | `(o bang)` | 0.682667 | OK |
| TABLE VII | `16.7` | TABLE VII token% strategyqa fixed_k1 | `(o bang)` | 16.6667 | OK |
| TABLE VII | `0.797` | TABLE VII accuracy gsm8k fixed_k2 | `(o bang)` | 0.796667 | OK |
| TABLE VII | `33.3` | TABLE VII token% gsm8k fixed_k2 | `(o bang)` | 33.3333 | OK |
| TABLE VII | `0.540` | TABLE VII accuracy mmlu fixed_k2 | `(o bang)` | 0.54 | OK |
| TABLE VII | `33.3` | TABLE VII token% mmlu fixed_k2 | `(o bang)` | 33.3333 | OK |
| TABLE VII | `0.698` | TABLE VII accuracy strategyqa fixed_k2 | `(o bang)` | 0.698 | OK |
| TABLE VII | `33.3` | TABLE VII token% strategyqa fixed_k2 | `(o bang)` | 33.3333 | OK |
| TABLE VII | `0.783` | TABLE VII accuracy gsm8k fixed_k3 | `(o bang)` | 0.782667 | OK |
| TABLE VII | `50.0` | TABLE VII token% gsm8k fixed_k3 | `(o bang)` | 50 | OK |
| TABLE VII | `0.550` | TABLE VII accuracy mmlu fixed_k3 | `(o bang)` | 0.55 | OK |
| TABLE VII | `50.0` | TABLE VII token% mmlu fixed_k3 | `(o bang)` | 50 | OK |
| TABLE VII | `0.691` | TABLE VII accuracy strategyqa fixed_k3 | `(o bang)` | 0.691333 | OK |
| TABLE VII | `50.0` | TABLE VII token% strategyqa fixed_k3 | `(o bang)` | 50 | OK |
| TABLE VII | `0.779` | TABLE VII accuracy gsm8k unc<q30 | `(o bang)` | 0.78 | **LECH** |
| TABLE VII | `70.7` | TABLE VII token% gsm8k unc<q30 | `(o bang)` | 70.6035 | **LECH** |
| TABLE VII | `0.548` | TABLE VII accuracy mmlu unc<q30 | `(o bang)` | 0.552667 | **LECH** |
| TABLE VII | `63.8` | TABLE VII token% mmlu unc<q30 | `(o bang)` | 62.793 | **LECH** |
| TABLE VII | `0.685` | TABLE VII accuracy strategyqa unc<q30 | `(o bang)` | 0.685333 | OK |
| TABLE VII | `68.1` | TABLE VII token% strategyqa unc<q30 | `(o bang)` | 67.4929 | **LECH** |
| TABLE VII | `0.781` | TABLE VII accuracy gsm8k unc<q40 | `(o bang)` | 0.782667 | **LECH** |
| TABLE VII | `63.0` | TABLE VII token% gsm8k unc<q40 | `(o bang)` | 62.6359 | **LECH** |
| TABLE VII | `0.543` | TABLE VII accuracy mmlu unc<q40 | `(o bang)` | 0.544667 | **LECH** |
| TABLE VII | `50.4` | TABLE VII token% mmlu unc<q40 | `(o bang)` | 48.9621 | **LECH** |
| TABLE VII | `0.687` | TABLE VII accuracy strategyqa unc<q40 | `(o bang)` | 0.686667 | OK |
| TABLE VII | `58.9` | TABLE VII token% strategyqa unc<q40 | `(o bang)` | 58.2885 | **LECH** |
| TABLE VII | `0.785` | TABLE VII accuracy gsm8k unc<q50 | `(o bang)` | 0.786 | **LECH** |
| TABLE VII | `52.6` | TABLE VII token% gsm8k unc<q50 | `(o bang)` | 52.1282 | **LECH** |
| TABLE VII | `0.545` | TABLE VII accuracy mmlu unc<q50 | `(o bang)` | 0.546667 | **LECH** |
| TABLE VII | `41.3` | TABLE VII token% mmlu unc<q50 | `(o bang)` | 40.1813 | **LECH** |
| TABLE VII | `0.690` | TABLE VII accuracy strategyqa unc<q50 | `(o bang)` | 0.694 | **LECH** |
| TABLE VII | `48.2` | TABLE VII token% strategyqa unc<q50 | `(o bang)` | 47.5 | **LECH** |
| TABLE VII | `0.863` | TABLE VII accuracy gsm8k oracle | `(o bang)` | 0.863333 | OK |
| TABLE VII | `32.7` | TABLE VII token% gsm8k oracle | `(o bang)` | 32.7029 | OK |
| TABLE VII | `0.681` | TABLE VII accuracy mmlu oracle | `(o bang)` | 0.681333 | OK |
| TABLE VII | `50.6` | TABLE VII token% mmlu oracle | `(o bang)` | 50.6195 | OK |
| TABLE VII | `0.787` | TABLE VII accuracy strategyqa oracle | `(o bang)` | 0.786667 | OK |
| TABLE VII | `38.0` | TABLE VII token% strategyqa oracle | `(o bang)` | 38.0207 | OK |
| TABLE VIII | `-0.001` | TABLE VIII gsm8k q0.5 dAcc mean | `(o bang)` | 0.000666667 | **LECH** |
| TABLE VIII | `0.004` | TABLE VIII gsm8k q0.5 dAcc sd | `(o bang)` | 0.00278887 | **LECH** |
| TABLE VIII | `-0.007` | TABLE VIII gsm8k q0.5 dAcc CI lo | `(o bang)` | -0.00678898 | OK |
| TABLE VIII | `0.006` | TABLE VIII gsm8k q0.5 dAcc CI hi | `(o bang)` | 0.00866133 | **LECH** |
| TABLE VIII | `-6.0` | TABLE VIII gsm8k q0.5 dToken mean | `(o bang)` | -6.84366 | **LECH** |
| TABLE VIII | `5.1` | TABLE VIII gsm8k q0.5 dToken sd | `(o bang)` | 4.70858 | **LECH** |
| TABLE VIII | `-0.003` | TABLE VIII mmlu q0.4 dAcc mean | `(o bang)` | -0.00133333 | **LECH** |
| TABLE VIII | `0.013` | TABLE VIII mmlu q0.4 dAcc sd | `(o bang)` | 0.0167664 | **LECH** |
| TABLE VIII | `-0.015` | TABLE VIII mmlu q0.4 dAcc CI lo | `(o bang)` | -0.0140028 | **LECH** |
| TABLE VIII | `0.008` | TABLE VIII mmlu q0.4 dAcc CI hi | `(o bang)` | 0.0110202 | **LECH** |
| TABLE VIII | `-26.1` | TABLE VIII mmlu q0.4 dToken mean | `(o bang)` | -28.2215 | **LECH** |
| TABLE VIII | `3.3` | TABLE VIII mmlu q0.4 dToken sd | `(o bang)` | 2.74953 | **LECH** |
| TABLE VIII | `-0.001` | TABLE VIII mmlu q0.5 dAcc mean | `(o bang)` | 0.000666667 | **LECH** |
| TABLE VIII | `0.013` | TABLE VIII mmlu q0.5 dAcc sd | `(o bang)` | 0.0192065 | **LECH** |
| TABLE VIII | `-0.015` | TABLE VIII mmlu q0.5 dAcc CI lo | `(o bang)` | -0.0138806 | **LECH** |
| TABLE VIII | `0.013` | TABLE VIII mmlu q0.5 dAcc CI hi | `(o bang)` | 0.0148968 | **LECH** |
| TABLE VIII | `-39.4` | TABLE VIII mmlu q0.5 dToken mean | `(o bang)` | -41.0658 | **LECH** |
| TABLE VIII | `2.3` | TABLE VIII mmlu q0.5 dToken sd | `(o bang)` | 2.19059 | **LECH** |
| TABLE VIII | `-0.001` | TABLE VIII strategyqa q0.5 dAcc mean | `(o bang)` | 0.00333333 | **LECH** |
| TABLE VIII | `0.008` | TABLE VIII strategyqa q0.5 dAcc sd | `(o bang)` | 0.00816497 | OK |
| TABLE VIII | `-0.009` | TABLE VIII strategyqa q0.5 dAcc CI lo | `(o bang)` | -0.005373 | **LECH** |
| TABLE VIII | `0.007` | TABLE VIII strategyqa q0.5 dAcc CI hi | `(o bang)` | 0.0119697 | **LECH** |
| TABLE VIII | `-0.4` | TABLE VIII strategyqa q0.5 dToken mean | `(o bang)` | -1.88818 | **LECH** |
| TABLE VIII | `8.3` | TABLE VIII strategyqa q0.5 dToken sd | `(o bang)` | 7.61559 | **LECH** |
| TABLE IX | `1500` | TABLE IX gsm8k n | `(o bang)` | 1500 | OK |
| TABLE IX | `54` | TABLE IX gsm8k rescued n | `(o bang)` | 54 | OK |
| TABLE IX | `3.6` | TABLE IX gsm8k rescued % | `(o bang)` | 3.6 | OK |
| TABLE IX | `62` | TABLE IX gsm8k corrupted n | `(o bang)` | 62 | OK |
| TABLE IX | `4.1` | TABLE IX gsm8k corrupted % | `(o bang)` | 4.13333 | OK |
| TABLE IX | `-8` | TABLE IX gsm8k net | `(o bang)` | -8 | OK |
| TABLE IX | `0.516` | TABLE IX gsm8k McNemar p | `(o bang)` | 0.515913 | OK |
| TABLE IX | `1500` | TABLE IX mmlu n | `(o bang)` | 1500 | OK |
| TABLE IX | `111` | TABLE IX mmlu rescued n | `(o bang)` | 111 | OK |
| TABLE IX | `7.4` | TABLE IX mmlu rescued % | `(o bang)` | 7.4 | OK |
| TABLE IX | `85` | TABLE IX mmlu corrupted n | `(o bang)` | 85 | OK |
| TABLE IX | `5.7` | TABLE IX mmlu corrupted % | `(o bang)` | 5.66667 | OK |
| TABLE IX | `26` | TABLE IX mmlu net | `(o bang)` | 26 | OK |
| TABLE IX | `0.074` | TABLE IX mmlu McNemar p | `(o bang)` | 0.0738758 | OK |
| TABLE IX | `1500` | TABLE IX strategyqa n | `(o bang)` | 1500 | OK |
| TABLE IX | `91` | TABLE IX strategyqa rescued n | `(o bang)` | 91 | OK |
| TABLE IX | `6.1` | TABLE IX strategyqa rescued % | `(o bang)` | 6.06667 | OK |
| TABLE IX | `86` | TABLE IX strategyqa corrupted n | `(o bang)` | 86 | OK |
| TABLE IX | `5.7` | TABLE IX strategyqa corrupted % | `(o bang)` | 5.73333 | OK |
| TABLE IX | `5` | TABLE IX strategyqa net | `(o bang)` | 5 | OK |
| TABLE IX | `0.764` | TABLE IX strategyqa McNemar p | `(o bang)` | 0.763774 | OK |
| TABLE IX | `4500` | TABLE IX total n | `(o bang)` | 4500 | OK |
| TABLE IX | `256` | TABLE IX total rescued n | `(o bang)` | 256 | OK |
| TABLE IX | `5.7` | TABLE IX total rescued % | `(o bang)` | 5.68889 | OK |
| TABLE IX | `233` | TABLE IX total corrupted n | `(o bang)` | 233 | OK |
| TABLE IX | `5.2` | TABLE IX total corrupted % | `(o bang)` | 5.17778 | OK |
| TABLE IX | `23` | TABLE IX total net | `(o bang)` | 23 | OK |
| TABLE IX | `0.320` | TABLE IX total McNemar p | `(o bang)` | 0.319798 | OK |
| TABLE X | `0.779` | TABLE X accuracy (always) gsm8k | `(o bang)` | 0.778667 | OK |
| TABLE X | `0.875` | TABLE X pooled AUC gsm8k | `(o bang)` | 0.874917 | OK |
| TABLE X | `56.0` | TABLE X consensus token% gsm8k | `(o bang)` | 55.9805 | OK |
| TABLE X | `-6.0` | TABLE X dToken achieved gsm8k | `(o bang)` | -6.84366 | **LECH** |
| TABLE X | `0.686` | TABLE X accuracy (always) strategyqa | `(o bang)` | 0.686 | OK |
| TABLE X | `0.609` | TABLE X pooled AUC strategyqa | `(o bang)` | 0.608987 | OK |
| TABLE X | `48.5` | TABLE X consensus token% strategyqa | `(o bang)` | 48.4502 | OK |
| TABLE X | `-0.4` | TABLE X dToken achieved strategyqa | `(o bang)` | -1.88818 | **LECH** |
| TABLE X | `0.547` | TABLE X accuracy (always) mmlu | `(o bang)` | 0.547333 | OK |
| TABLE X | `0.685` | TABLE X pooled AUC mmlu | `(o bang)` | 0.68509 | OK |
| TABLE X | `68.2` | TABLE X consensus token% mmlu | `(o bang)` | 68.21 | OK |
| TABLE X | `-39.4` | TABLE X dToken achieved mmlu | `(o bang)` | -41.0658 | **LECH** |
| TABLE 0 | `0.779` | TABLE 0 accuracy gsm8k always | `(o bang)` | 0.778667 | OK |
| TABLE 0 | `100.0` | TABLE 0 token% gsm8k always | `(o bang)` | 100 | OK |
| TABLE 0 | `0.547` | TABLE 0 accuracy mmlu always | `(o bang)` | 0.547333 | OK |
| TABLE 0 | `100.0` | TABLE 0 token% mmlu always | `(o bang)` | 100 | OK |
| TABLE 0 | `0.686` | TABLE 0 accuracy strategyqa always | `(o bang)` | 0.686 | OK |
| TABLE 0 | `100.0` | TABLE 0 token% strategyqa always | `(o bang)` | 100 | OK |
| TABLE 0 | `0.863` | TABLE 0 accuracy gsm8k oracle | `(o bang)` | 0.863333 | OK |
| TABLE 0 | `32.7` | TABLE 0 token% gsm8k oracle | `(o bang)` | 32.7061 | OK |
| TABLE 0 | `0.681` | TABLE 0 accuracy mmlu oracle | `(o bang)` | 0.681333 | OK |
| TABLE 0 | `50.7` | TABLE 0 token% mmlu oracle | `(o bang)` | 50.6821 | OK |
| TABLE 0 | `0.787` | TABLE 0 accuracy strategyqa oracle | `(o bang)` | 0.786667 | OK |
| TABLE 0 | `38.0` | TABLE 0 token% strategyqa oracle | `(o bang)` | 38.0141 | OK |
| TABLE 0 | `0.762` | TABLE 0 accuracy gsm8k ensemble_vote | `(o bang)` | 0.762 | OK |
| TABLE 0 | `11.2` | TABLE 0 token% gsm8k ensemble_vote | `(o bang)` | 11.1543 | OK |
| TABLE 0 | `0.521` | TABLE 0 accuracy mmlu ensemble_vote | `(o bang)` | 0.520667 | OK |
| TABLE 0 | `13.2` | TABLE 0 token% mmlu ensemble_vote | `(o bang)` | 13.219 | OK |
| TABLE 0 | `0.615` | TABLE 0 accuracy strategyqa ensemble_vote | `(o bang)` | 0.615333 | OK |
| TABLE 0 | `5.6` | TABLE 0 token% strategyqa ensemble_vote | `(o bang)` | 5.57619 | OK |
| TABLE 0 | `0.638` | TABLE 0 accuracy gsm8k self_consistency_a | `(o bang)` | 0.638 | OK |
| TABLE 0 | `10.9` | TABLE 0 token% gsm8k self_consistency_a | `(o bang)` | 10.8716 | OK |
| TABLE 0 | `0.582` | TABLE 0 accuracy mmlu self_consistency_a | `(o bang)` | 0.582 | OK |
| TABLE 0 | `11.1` | TABLE 0 token% mmlu self_consistency_a | `(o bang)` | 11.1126 | OK |
| TABLE 0 | `0.548` | TABLE 0 accuracy strategyqa self_consistency_a | `(o bang)` | 0.548 | OK |
| TABLE 0 | `4.1` | TABLE 0 token% strategyqa self_consistency_a | `(o bang)` | 4.08522 | OK |
| TABLE 0 | `0.691` | TABLE 0 accuracy gsm8k self_consistency_b | `(o bang)` | 0.690667 | OK |
| TABLE 0 | `9.4` | TABLE 0 token% gsm8k self_consistency_b | `(o bang)` | 9.35001 | OK |
| TABLE 0 | `0.293` | TABLE 0 accuracy mmlu self_consistency_b | `(o bang)` | 0.293333 | OK |
| TABLE 0 | `16.4` | TABLE 0 token% mmlu self_consistency_b | `(o bang)` | 16.3997 | OK |
| TABLE 0 | `0.611` | TABLE 0 accuracy strategyqa self_consistency_b | `(o bang)` | 0.611333 | OK |
| TABLE 0 | `4.1` | TABLE 0 token% strategyqa self_consistency_b | `(o bang)` | 4.11333 | OK |
| TABLE 0 | `0.780` | TABLE 0 accuracy gsm8k self_consistency_c | `(o bang)` | 0.78 | OK |
| TABLE 0 | `12.7` | TABLE 0 token% gsm8k self_consistency_c | `(o bang)` | 12.6634 | OK |
| TABLE 0 | `0.515` | TABLE 0 accuracy mmlu self_consistency_c | `(o bang)` | 0.514667 | OK |
| TABLE 0 | `10.5` | TABLE 0 token% mmlu self_consistency_c | `(o bang)` | 10.4812 | OK |
| TABLE 0 | `0.679` | TABLE 0 accuracy strategyqa self_consistency_c | `(o bang)` | 0.678667 | OK |
| TABLE 0 | `8.4` | TABLE 0 token% strategyqa self_consistency_c | `(o bang)` | 8.41793 | OK |
| TABLE 0 | `0.784` | TABLE 0 accuracy gsm8k fixed_k1 | `(o bang)` | 0.784 | OK |
| TABLE 0 | `16.7` | TABLE 0 token% gsm8k fixed_k1 | `(o bang)` | 16.6667 | OK |
| TABLE 0 | `0.530` | TABLE 0 accuracy mmlu fixed_k1 | `(o bang)` | 0.53 | OK |
| TABLE 0 | `16.7` | TABLE 0 token% mmlu fixed_k1 | `(o bang)` | 16.6667 | OK |
| TABLE 0 | `0.683` | TABLE 0 accuracy strategyqa fixed_k1 | `(o bang)` | 0.682667 | OK |
| TABLE 0 | `16.7` | TABLE 0 token% strategyqa fixed_k1 | `(o bang)` | 16.6667 | OK |
| TABLE 0 | `0.797` | TABLE 0 accuracy gsm8k fixed_k2 | `(o bang)` | 0.796667 | OK |
| TABLE 0 | `33.3` | TABLE 0 token% gsm8k fixed_k2 | `(o bang)` | 33.3333 | OK |
| TABLE 0 | `0.540` | TABLE 0 accuracy mmlu fixed_k2 | `(o bang)` | 0.54 | OK |
| TABLE 0 | `33.3` | TABLE 0 token% mmlu fixed_k2 | `(o bang)` | 33.3333 | OK |
| TABLE 0 | `0.698` | TABLE 0 accuracy strategyqa fixed_k2 | `(o bang)` | 0.698 | OK |
| TABLE 0 | `33.3` | TABLE 0 token% strategyqa fixed_k2 | `(o bang)` | 33.3333 | OK |
| TABLE 0 | `0.783` | TABLE 0 accuracy gsm8k fixed_k3 | `(o bang)` | 0.782667 | OK |
| TABLE 0 | `50.0` | TABLE 0 token% gsm8k fixed_k3 | `(o bang)` | 50 | OK |
| TABLE 0 | `0.550` | TABLE 0 accuracy mmlu fixed_k3 | `(o bang)` | 0.55 | OK |
| TABLE 0 | `50.0` | TABLE 0 token% mmlu fixed_k3 | `(o bang)` | 50 | OK |
| TABLE 0 | `0.691` | TABLE 0 accuracy strategyqa fixed_k3 | `(o bang)` | 0.691333 | OK |
| TABLE 0 | `50.0` | TABLE 0 token% strategyqa fixed_k3 | `(o bang)` | 50 | OK |
| TABLE 0 | `0.779` | TABLE 0 accuracy gsm8k fixed_k4 | `(o bang)` | 0.779333 | OK |
| TABLE 0 | `66.7` | TABLE 0 token% gsm8k fixed_k4 | `(o bang)` | 66.6667 | OK |
| TABLE 0 | `0.557` | TABLE 0 accuracy mmlu fixed_k4 | `(o bang)` | 0.556667 | OK |
| TABLE 0 | `66.7` | TABLE 0 token% mmlu fixed_k4 | `(o bang)` | 66.6667 | OK |
| TABLE 0 | `0.677` | TABLE 0 accuracy strategyqa fixed_k4 | `(o bang)` | 0.677333 | OK |
| TABLE 0 | `66.7` | TABLE 0 token% strategyqa fixed_k4 | `(o bang)` | 66.6667 | OK |
| TABLE 0 | `0.793` | TABLE 0 accuracy gsm8k fixed_k5 | `(o bang)` | 0.793333 | OK |
| TABLE 0 | `83.3` | TABLE 0 token% gsm8k fixed_k5 | `(o bang)` | 83.3333 | OK |
| TABLE 0 | `0.563` | TABLE 0 accuracy mmlu fixed_k5 | `(o bang)` | 0.563333 | OK |
| TABLE 0 | `83.3` | TABLE 0 token% mmlu fixed_k5 | `(o bang)` | 83.3333 | OK |
| TABLE 0 | `0.689` | TABLE 0 accuracy strategyqa fixed_k5 | `(o bang)` | 0.688667 | OK |
| TABLE 0 | `83.3` | TABLE 0 token% strategyqa fixed_k5 | `(o bang)` | 83.3333 | OK |
| TABLE 0 | `0.785` | TABLE 0 accuracy gsm8k consensus | `(o bang)` | 0.785333 | OK |
| TABLE 0 | `56.0` | TABLE 0 token% gsm8k consensus | `(o bang)` | 55.9798 | OK |
| TABLE 0 | `0.546` | TABLE 0 accuracy mmlu consensus | `(o bang)` | 0.546 | OK |
| TABLE 0 | `68.2` | TABLE 0 token% mmlu consensus | `(o bang)` | 68.203 | OK |
| TABLE 0 | `0.691` | TABLE 0 accuracy strategyqa consensus | `(o bang)` | 0.690667 | OK |
| TABLE 0 | `48.5` | TABLE 0 token% strategyqa consensus | `(o bang)` | 48.4575 | OK |
| TABLE 0 | `0.781` | TABLE 0 accuracy gsm8k unc<q40 | `(o bang)` | 0.782667 | **LECH** |
| TABLE 0 | `63.0` | TABLE 0 token% gsm8k unc<q40 | `(o bang)` | 62.6437 | **LECH** |
| TABLE 0 | `0.543` | TABLE 0 accuracy mmlu unc<q40 | `(o bang)` | 0.544667 | **LECH** |
| TABLE 0 | `50.4` | TABLE 0 token% mmlu unc<q40 | `(o bang)` | 48.9053 | **LECH** |
| TABLE 0 | `0.687` | TABLE 0 accuracy strategyqa unc<q40 | `(o bang)` | 0.686667 | OK |
| TABLE 0 | `58.9` | TABLE 0 token% strategyqa unc<q40 | `(o bang)` | 58.295 | **LECH** |
| TABLE 0 | `0.785` | TABLE 0 accuracy gsm8k unc<q50 | `(o bang)` | 0.786 | **LECH** |
| TABLE 0 | `52.6` | TABLE 0 token% gsm8k unc<q50 | `(o bang)` | 52.1522 | **LECH** |
| TABLE 0 | `0.545` | TABLE 0 accuracy mmlu unc<q50 | `(o bang)` | 0.546667 | **LECH** |
| TABLE 0 | `41.3` | TABLE 0 token% mmlu unc<q50 | `(o bang)` | 40.1685 | **LECH** |
| TABLE 0 | `0.690` | TABLE 0 accuracy strategyqa unc<q50 | `(o bang)` | 0.694 | **LECH** |
| TABLE 0 | `48.2` | TABLE 0 token% strategyqa unc<q50 | `(o bang)` | 47.5019 | **LECH** |
| TABLE XI | `0.0153` | TABLE XI dacc gsm8k unc<q40 | `(o bang)` | 0.014 | **LECH** |
| TABLE XI | `0.0013` | TABLE XI CI-lo gsm8k unc<q40 | `(o bang)` | 0 | **LECH** |
| TABLE XI | `0.0293` | TABLE XI CI-hi gsm8k unc<q40 | `(o bang)` | 0.0278146 | **LECH** |
| TABLE XI | `0.035` | TABLE XI p gsm8k unc<q40 | `(o bang)` | 0.0514 | **LECH** |
| TABLE XI | `-29.7` | TABLE XI dtok gsm8k unc<q40 | `(o bang)` | -29.3103 | **LECH** |
| TABLE XI | `0.0120` | TABLE XI dacc gsm8k unc<q50 | `(o bang)` | 0.0106667 | **LECH** |
| TABLE XI | `-0.0013` | TABLE XI CI-lo gsm8k unc<q50 | `(o bang)` | -0.00267023 | **LECH** |
| TABLE XI | `0.0253` | TABLE XI CI-hi gsm8k unc<q50 | `(o bang)` | 0.0238573 | **LECH** |
| TABLE XI | `0.080` | TABLE XI p gsm8k unc<q50 | `(o bang)` | 0.1286 | **LECH** |
| TABLE XI | `-19.3` | TABLE XI dtok gsm8k unc<q50 | `(o bang)` | -18.8189 | **LECH** |
| TABLE XI | `-0.0027` | TABLE XI dacc mmlu unc<q40 | `(o bang)` | -0.00466667 | **LECH** |
| TABLE XI | `-0.0182` | TABLE XI CI-lo mmlu unc<q40 | `(o bang)` | -0.0208344 | **LECH** |
| TABLE XI | `0.0132` | TABLE XI CI-hi mmlu unc<q40 | `(o bang)` | 0.012008 | **LECH** |
| TABLE XI | `0.768` | TABLE XI p mmlu unc<q40 | `(o bang)` | 0.5996 | **LECH** |
| TABLE XI | `-17.0` | TABLE XI dtok mmlu unc<q40 | `(o bang)` | -15.572 | **LECH** |
| TABLE XI | `-0.0053` | TABLE XI dacc mmlu unc<q50 | `(o bang)` | -0.00666667 | **LECH** |
| TABLE XI | `-0.0207` | TABLE XI CI-lo mmlu unc<q50 | `(o bang)` | -0.023118 | **LECH** |
| TABLE XI | `0.0101` | TABLE XI CI-hi mmlu unc<q50 | `(o bang)` | 0.0100876 | OK |
| TABLE XI | `0.525` | TABLE XI p mmlu unc<q50 | `(o bang)` | 0.4656 | **LECH** |
| TABLE XI | `-8.0` | TABLE XI dtok mmlu unc<q50 | `(o bang)` | -6.8352 | **LECH** |
| TABLE XI | `0.0107` | TABLE XI dacc strategyqa unc<q40 | `(o bang)` | 0.0113333 | **LECH** |
| TABLE XI | `-0.0046` | TABLE XI CI-lo strategyqa unc<q40 | `(o bang)` | -0.00406787 | **LECH** |
| TABLE XI | `0.0257` | TABLE XI CI-hi strategyqa unc<q40 | `(o bang)` | 0.0267739 | **LECH** |
| TABLE XI | `0.178` | TABLE XI p strategyqa unc<q40 | `(o bang)` | 0.1626 | **LECH** |
| TABLE XI | `-25.6` | TABLE XI dtok strategyqa unc<q40 | `(o bang)` | -24.9617 | **LECH** |
| TABLE XI | `0.0080` | TABLE XI dacc strategyqa unc<q50 | `(o bang)` | 0.004 | **LECH** |
| TABLE XI | `-0.0066` | TABLE XI CI-lo strategyqa unc<q50 | `(o bang)` | -0.0104923 | **LECH** |
| TABLE XI | `0.0226` | TABLE XI CI-hi strategyqa unc<q50 | `(o bang)` | 0.0187668 | **LECH** |
| TABLE XI | `0.310` | TABLE XI p strategyqa unc<q50 | `(o bang)` | 0.6394 | **LECH** |
| TABLE XI | `-14.9` | TABLE XI dtok strategyqa unc<q50 | `(o bang)` | -14.1686 | **LECH** |
| TABLE XII | `-0.0227` | TABLE XII dacc gsm8k ensemble_vote | `(o bang)` | -0.024 | **LECH** |
| TABLE XII | `-0.0419` | TABLE XII CI-lo gsm8k ensemble_vote | `(o bang)` | -0.043166 | **LECH** |
| TABLE XII | `-0.0038` | TABLE XII CI-hi gsm8k ensemble_vote | `(o bang)` | -0.00470091 | **LECH** |
| TABLE XII | `0.024` | TABLE XII p gsm8k ensemble_vote | `(o bang)` | 0.0178 | **LECH** |
| TABLE XII | `-41.5` | TABLE XII dtok gsm8k ensemble_vote | `(o bang)` | -40.9979 | **LECH** |
| TABLE XII | `-0.1467` | TABLE XII dacc gsm8k self_consistency_a | `(o bang)` | -0.148 | **LECH** |
| TABLE XII | `-0.1752` | TABLE XII CI-lo gsm8k self_consistency_a | `(o bang)` | -0.176628 | **LECH** |
| TABLE XII | `-0.1178` | TABLE XII CI-hi gsm8k self_consistency_a | `(o bang)` | -0.119218 | **LECH** |
| TABLE XII | `0.0001` | TABLE XII p gsm8k self_consistency_a | `(o bang)` | 0.0001 | OK |
| TABLE XII | `-41.8` | TABLE XII dtok gsm8k self_consistency_a | `(o bang)` | -41.2806 | **LECH** |
| TABLE XII | `-0.0940` | TABLE XII dacc gsm8k self_consistency_b | `(o bang)` | -0.0953333 | **LECH** |
| TABLE XII | `-0.1210` | TABLE XII CI-lo gsm8k self_consistency_b | `(o bang)` | -0.122245 | **LECH** |
| TABLE XII | `-0.0668` | TABLE XII CI-hi gsm8k self_consistency_b | `(o bang)` | -0.0679806 | **LECH** |
| TABLE XII | `0.0001` | TABLE XII p gsm8k self_consistency_b | `(o bang)` | 0.0001 | OK |
| TABLE XII | `-43.3` | TABLE XII dtok gsm8k self_consistency_b | `(o bang)` | -42.8022 | **LECH** |
| TABLE XII | `-0.0047` | TABLE XII dacc gsm8k self_consistency_c | `(o bang)` | -0.006 | **LECH** |
| TABLE XII | `-0.0224` | TABLE XII CI-lo gsm8k self_consistency_c | `(o bang)` | -0.0233615 | **LECH** |
| TABLE XII | `0.0132` | TABLE XII CI-hi gsm8k self_consistency_c | `(o bang)` | 0.0117045 | **LECH** |
| TABLE XII | `0.630` | TABLE XII p gsm8k self_consistency_c | `(o bang)` | 0.5232 | **LECH** |
| TABLE XII | `-40.0` | TABLE XII dtok gsm8k self_consistency_c | `(o bang)` | -39.4888 | **LECH** |
| TABLE XII | `-0.0247` | TABLE XII dacc mmlu ensemble_vote | `(o bang)` | -0.026 | **LECH** |
| TABLE XII | `-0.0493` | TABLE XII CI-lo mmlu ensemble_vote | `(o bang)` | -0.0506581 | **LECH** |
| TABLE XII | `0.0007` | TABLE XII CI-hi mmlu ensemble_vote | `(o bang)` | -0.00132888 | **LECH** |
| TABLE XII | `0.055` | TABLE XII p mmlu ensemble_vote | `(o bang)` | 0.0432 | **LECH** |
| TABLE XII | `-28.1` | TABLE XII dtok mmlu ensemble_vote | `(o bang)` | -26.9495 | **LECH** |
| TABLE XII | `0.0367` | TABLE XII dacc mmlu self_consistency_a | `(o bang)` | 0.0353333 | **LECH** |
| TABLE XII | `0.0072` | TABLE XII CI-lo mmlu self_consistency_a | `(o bang)` | 0.0059997 | **LECH** |
| TABLE XII | `0.0661` | TABLE XII CI-hi mmlu self_consistency_a | `(o bang)` | 0.064621 | **LECH** |
| TABLE XII | `0.015` | TABLE XII p mmlu self_consistency_a | `(o bang)` | 0.0186 | **LECH** |
| TABLE XII | `-30.2` | TABLE XII dtok mmlu self_consistency_a | `(o bang)` | -29.0559 | **LECH** |
| TABLE XII | `-0.2520` | TABLE XII dacc mmlu self_consistency_b | `(o bang)` | -0.253333 | **LECH** |
| TABLE XII | `-0.2882` | TABLE XII CI-lo mmlu self_consistency_b | `(o bang)` | -0.289913 | **LECH** |
| TABLE XII | `-0.2161` | TABLE XII CI-hi mmlu self_consistency_b | `(o bang)` | -0.216689 | **LECH** |
| TABLE XII | `0.0001` | TABLE XII p mmlu self_consistency_b | `(o bang)` | 0.0001 | OK |
| TABLE XII | `-24.9` | TABLE XII dtok mmlu self_consistency_b | `(o bang)` | -23.7688 | **LECH** |
| TABLE XII | `-0.0307` | TABLE XII dacc mmlu self_consistency_c | `(o bang)` | -0.032 | **LECH** |
| TABLE XII | `-0.0521` | TABLE XII CI-lo mmlu self_consistency_c | `(o bang)` | -0.0524597 | OK |
| TABLE XII | `-0.0092` | TABLE XII CI-hi mmlu self_consistency_c | `(o bang)` | -0.0113103 | **LECH** |
| TABLE XII | `0.006` | TABLE XII p mmlu self_consistency_c | `(o bang)` | 0.0028 | **LECH** |
| TABLE XII | `-30.8` | TABLE XII dtok mmlu self_consistency_c | `(o bang)` | -29.6874 | **LECH** |
| TABLE XII | `-0.0747` | TABLE XII dacc strategyqa ensemble_vote | `(o bang)` | -0.0786667 | **LECH** |
| TABLE XII | `-0.1077` | TABLE XII CI-lo strategyqa ensemble_vote | `(o bang)` | -0.111333 | **LECH** |
| TABLE XII | `-0.0405` | TABLE XII CI-hi strategyqa ensemble_vote | `(o bang)` | -0.045097 | **LECH** |
| TABLE XII | `0.0002` | TABLE XII p strategyqa ensemble_vote | `(o bang)` | 0.0001 | OK |
| TABLE XII | `-42.6` | TABLE XII dtok strategyqa ensemble_vote | `(o bang)` | -41.9258 | **LECH** |
| TABLE XII | `-0.1420` | TABLE XII dacc strategyqa self_consistency_a | `(o bang)` | -0.146 | **LECH** |
| TABLE XII | `-0.1880` | TABLE XII CI-lo strategyqa self_consistency_a | `(o bang)` | -0.19163 | **LECH** |
| TABLE XII | `-0.0961` | TABLE XII CI-hi strategyqa self_consistency_a | `(o bang)` | -0.0999351 | **LECH** |
| TABLE XII | `0.0001` | TABLE XII p strategyqa self_consistency_a | `(o bang)` | 0.0001 | OK |
| TABLE XII | `-44.1` | TABLE XII dtok strategyqa self_consistency_a | `(o bang)` | -43.4167 | **LECH** |
| TABLE XII | `-0.0787` | TABLE XII dacc strategyqa self_consistency_b | `(o bang)` | -0.0826667 | **LECH** |
| TABLE XII | `-0.1165` | TABLE XII CI-lo strategyqa self_consistency_b | `(o bang)` | -0.120395 | **LECH** |
| TABLE XII | `-0.0411` | TABLE XII CI-hi strategyqa self_consistency_b | `(o bang)` | -0.045422 | **LECH** |
| TABLE XII | `0.0002` | TABLE XII p strategyqa self_consistency_b | `(o bang)` | 0.0001 | OK |
| TABLE XII | `-44.1` | TABLE XII dtok strategyqa self_consistency_b | `(o bang)` | -43.3886 | **LECH** |
| TABLE XII | `-0.0113` | TABLE XII dacc strategyqa self_consistency_c | `(o bang)` | -0.0153333 | **LECH** |
| TABLE XII | `-0.0317` | TABLE XII CI-lo strategyqa self_consistency_c | `(o bang)` | -0.0352163 | **LECH** |
| TABLE XII | `0.0090` | TABLE XII CI-hi strategyqa self_consistency_c | `(o bang)` | 0.00465124 | **LECH** |
| TABLE XII | `0.293` | TABLE XII p strategyqa self_consistency_c | `(o bang)` | 0.1434 | **LECH** |
| TABLE XII | `-39.8` | TABLE XII dtok strategyqa self_consistency_c | `(o bang)` | -39.084 | **LECH** |
| V-B | `27000` | so round dang thuc entropy=0 <=> consensus dung | `LIVE: dem round co (answer_entropy == 0) == consensus_now  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 27000 | OK |
| VI-A | `978` | so cau hoi DUY NHAT tren gsm8k (1500 luot rut) | `LIVE: nunique(sample_id) tren gsm8k  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 978 | OK |
| VI-A | `1123` | so cau hoi DUY NHAT tren mmlu (1500 luot rut) | `LIVE: nunique(sample_id) tren mmlu  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 1123 | OK |
| VI-A | `647` | so cau hoi DUY NHAT tren strategyqa (1500 luot rut) | `LIVE: nunique(sample_id) tren strategyqa  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 647 | OK |
| VI-A | `46` | chong lan cau hoi lon nhat giua hai seed bat ky (%) | `LIVE: max |giao| / 300 tren moi cap seed, moi benchmark  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 46 | OK |
| VI-A | `12` | chong lan cau hoi nho nhat giua hai seed bat ky (%) | `LIVE: min |giao| / 300 tren moi cap seed, moi benchmark  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 12 | OK |
| IV-D | `0.992` | do giong prompt solver(r0) vs independent tren gsm8k | `LIVE: SequenceMatcher.ratio(), cau hoi thay bang 'WHAT_IS_THE_QUESTION'  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.992124 | OK |
| IV-D | `0.994` | do giong prompt solver(r0) vs independent tren mmlu | `LIVE: SequenceMatcher.ratio(), cau hoi thay bang 'WHAT_IS_THE_QUESTION'  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.994404 | OK |
| IV-D | `0.027` | do giong prompt solver(r0) vs independent tren strategyqa | `LIVE: SequenceMatcher.ratio(), cau hoi thay bang 'WHAT_IS_THE_QUESTION'  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.0266042 | OK |
| IV-D | `2.7` | do giong prompt tren strategyqa, dang phan tram | `LIVE: sim(strategyqa) * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.0266042 | OK |
| IV-D | `8390` | do dai prompt solver(r0) tren strategyqa | `LIVE: len(build_solver_prompt), cau hoi thay bang 'WHAT_IS_THE_QUESTION'  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 8390 | OK |
| IV-D | `1007` | do dai prompt independent tren strategyqa | `LIVE: len(build_independent_prompt), cau hoi thay bang 'WHAT_IS_THE_QUESTION'  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 1007 | OK |
| V | `6.7` | chenh lech lon nhat giua fixed_k1 va majority voting chay rieng (diem) | `LIVE: max |acc(fixed_k1) - acc(ensemble_vote)| * 100 qua 3 benchmark  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 6.73333 | OK |
| VI-C | `1701` | so dap an MMLU nam ngoai A-D | `LIVE: dem tren log debate  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 1701 | OK |
| VI-C | `6.30` | ty le dap an MMLU khong parse duoc | `LIVE: 1701 / 27000  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 6.3 | OK |
| VI-C | `13.4` | ty le khong parse duoc tren MMLU, solver_b | `LIVE: 1210 / 9000  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 13.4444 | OK |
| VI-C | `2.5` | ty le khong parse duoc tren MMLU, solver_a | `LIVE: 221 / 9000  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 2.45556 | OK |
| VI-C | `3.0` | ty le khong parse duoc tren MMLU, critic | `LIVE: 270 / 9000  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 3 | OK |
| VI-C | `0.00` | ty le khong parse duoc tren GSM8K va StrategyQA | `LIVE: dem tren log debate, ca hai benchmark  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0 | OK |
| VI-A | `3.7` | worst-case regret cua unc<q50 (diem accuracy) | `LIVE: max qua benchmark cua (acc tot nhat - acc chinh sach) * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 3.53333 | **LECH** |
| VI-A | `4.2` | worst-case regret cua fixed_k2 (diem accuracy) | `LIVE: max qua benchmark cua (acc tot nhat - acc chinh sach) * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 4.2 | OK |
| VI-A | `6.7` | worst-case regret cua self_consistency_c (diem accuracy) | `LIVE: max qua benchmark cua (acc tot nhat - acc chinh sach) * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 6.73333 | OK |
| VI-A | `8.3` | worst-case regret cua ensemble_vote (diem accuracy) | `LIVE: max qua benchmark cua (acc tot nhat - acc chinh sach) * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 8.26667 | OK |
| VI-A | `15.9` | worst-case regret cua self_consistency_a (diem accuracy) | `LIVE: max qua benchmark cua (acc tot nhat - acc chinh sach) * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 15.8667 | OK |
| V | `29` | do trai rong lon nhat giua ba model self-consistency tren mot benchmark | `LIVE: max - min acc cua SC a/b/c tren MMLU, * 100  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 28.8667 | OK |
| V-A | `20385` | so vong khong-cuoi chot cung ket cuc voi vong cuoi | `LIVE: voi moi round r < round cuoi: so ok_if_stop[r] voi ok_if_stop[cuoi]  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 20385 | OK |
| V-A | `90.6` | ty le vong khong-cuoi khong doi ket cuc | `LIVE: 20,385 / 22,500  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.906 | OK |
| V-A | `11754` | so round co consensus | `LIVE: dem round consensus_now == True  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 11754 | OK |
| V-A | `2020` | so round consensus nhung chot sai | `LIVE: trong round consensus, dem ok_if_stop == False  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 2020 | OK |
| V-A | `17.2` | blind-spot rate chung | `LIVE: 2,020 / 11,754  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.171856 | OK |
| TABLE I | `2.9` | blind-spot rate gsm8k | `LIVE: trong round consensus cua gsm8k, ty le ok_if_stop == False  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.0288833 | OK |
| TABLE I | `28.0` | blind-spot rate mmlu | `LIVE: trong round consensus cua mmlu, ty le ok_if_stop == False  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.280395 | OK |
| TABLE I | `24.4` | blind-spot rate strategyqa | `LIVE: trong round consensus cua strategyqa, ty le ok_if_stop == False  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.244444 | OK |
| IV-C | `685` | so cau hoi split train cua gsm8k | `LIVE: dem sample_id trong results/split/gsm8k/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 685 | OK |
| IV-C | `196` | so cau hoi split val cua gsm8k | `LIVE: dem sample_id trong results/split/gsm8k/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 196 | OK |
| IV-C | `97` | so cau hoi split test cua gsm8k | `LIVE: dem sample_id trong results/split/gsm8k/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 97 | OK |
| IV-C | `786` | so cau hoi split train cua mmlu | `LIVE: dem sample_id trong results/split/mmlu/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 786 | OK |
| IV-C | `225` | so cau hoi split val cua mmlu | `LIVE: dem sample_id trong results/split/mmlu/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 225 | OK |
| IV-C | `112` | so cau hoi split test cua mmlu | `LIVE: dem sample_id trong results/split/mmlu/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 112 | OK |
| IV-C | `453` | so cau hoi split train cua strategyqa | `LIVE: dem sample_id trong results/split/strategyqa/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 453 | OK |
| IV-C | `129` | so cau hoi split val cua strategyqa | `LIVE: dem sample_id trong results/split/strategyqa/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 129 | OK |
| IV-C | `65` | so cau hoi split test cua strategyqa | `LIVE: dem sample_id trong results/split/strategyqa/split_sample_ids.json  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 65 | OK |
| V-E | `0.547` | accuracy cua `always` tren toan bo MMLU | `LIVE: trong moi debate mmlu: trung binh theo seed cua ty le dung o round cuoi  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.547333 | OK |
| V-E | `0.587` | accuracy cua `always` tren tap test MMLU | `LIVE: trong debate mmlu thuoc split test: trung binh theo seed cua ty le dung o round cuoi  [results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)]` | 0.586789 | OK |

## Numbers in the paper not covered above

`0.00185`, `0.0033`, `0.5473`, `0.76`, `10000`, `14.2`, `14.7`, `25.2`, `27`