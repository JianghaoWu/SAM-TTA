
# python adaptation_nonlinear_wjh_22.py --cfg configs/config_brats.py --dataset BraTS_PED_t2w_2D --prompt box --gpu_ids "1"
# python adaptation_nonlinear_wjh_22.py --cfg configs/config_brats.py --dataset BraTS_PED_t2f_2D --prompt box --gpu_ids "1"
# python adaptation_nonlinear_wjh_22.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2w_2D --prompt box --gpu_ids "1"
# python adaptation_nonlinear_wjh_22.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2f_2D --prompt box --gpu_ids "1"

python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset mbh --prompt box --gpu_ids "1"
python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset BraTS_PED_t2w_2D --prompt box --gpu_ids "1"
python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset BraTS_PED_t2f_2D --prompt box --gpu_ids "1"
python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2w_2D --prompt box --gpu_ids "1"
python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2f_2D --prompt box --gpu_ids "1"
python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset CVC --prompt box --gpu_ids "1"
python adaptation_medsam_ttaa.py --cfg configs/config_brats.py --dataset PraNet --prompt box --gpu_ids "1"