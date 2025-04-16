# python adaptation_nonlinear_wjh.py --cfg configs/config_brats.py --dataset mbh --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh.py --cfg configs/config_brats.py --dataset BraTS_PED_t2w_2D --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh.py --cfg configs/config_brats.py --dataset BraTS_PED_t2f_2D --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2w_2D --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2f_2D --prompt box --gpu_ids "0"
# python adaptation_cotta.py --cfg configs/config_brats.py --dataset mbh --prompt box --gpu_ids "0"
# python adaptation_cotta.py --cfg configs/config_brats.py --dataset BraTS_PED_t2w_2D --prompt box --gpu_ids "0"
# python adaptation_cotta.py --cfg configs/config_brats.py --dataset BraTS_PED_t2f_2D --prompt box --gpu_ids "0"
# python adaptation_cotta.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2w_2D --prompt box --gpu_ids "0"
# python adaptation_cotta.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2f_2D --prompt box --gpu_ids "0"

# python adaptation_nonlinear_wjh_RGB.py --cfg configs/config_brats.py --dataset CVC --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh_RGB.py --cfg configs/config_brats.py --dataset PraNet --prompt box --gpu_ids "0"

# python adaptation_nonlinear_wjh_3.py --cfg configs/config_brats.py --dataset mbh --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh_3.py --cfg configs/config_brats.py --dataset BraTS_PED_t2w_2D --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh_3.py --cfg configs/config_brats.py --dataset BraTS_PED_t2f_2D --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh_3.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2w_2D --prompt box --gpu_ids "0"
# python adaptation_nonlinear_wjh_3.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2f_2D --prompt box --gpu_ids "0"


python adaptation_wesam.py --cfg configs/config_brats.py --dataset mbh --prompt box --gpu_ids "0"

python adaptation_upl.py --cfg configs/config_brats.py --dataset mbh --prompt box --gpu_ids "1"

python adaptation_medsam_tta.py --cfg configs/config_brats.py --dataset BraTS_PED_t2w_2D --prompt box --gpu_ids "0"
python adaptation_medsam_tta.py --cfg configs/config_brats.py --dataset BraTS_PED_t2f_2D --prompt box --gpu_ids "0"
python adaptation_medsam_tta.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2w_2D --prompt box --gpu_ids "0"
python adaptation_medsam_tta.py --cfg configs/config_brats.py --dataset BraTS_SSA_t2f_2D --prompt box --gpu_ids "0"
python adaptation_medsam_tta.py --cfg configs/config_brats.py --dataset CVC --prompt box --gpu_ids "0"
python adaptation_medsam_tta.py --cfg configs/config_brats.py --dataset PraNet --prompt box --gpu_ids "0"