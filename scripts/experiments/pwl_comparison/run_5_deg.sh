# PWL
cd ~/work/vnn/PWL-Geometric-Verification/
python ./main.py --bound_type pw_linear --dset MNIST --image_number 101 --transformation rotate --LB 0 --UB 5 --save_bounds
python ./main.py --bound_type pw_linear --dset CIFAR --image_number 101 --transformation rotate --LB 0 --UB 5 --save_bounds

# Ours
# num_init_splits=124
python ./scripts/calculate_bounds.py --dataset MNIST --image-number 100 --padding-value 0 --lipschitz-error 0.05 --max-bab-iter 5000 --num-jobs=-1 --transformation ROTATE --lower 0 --upper 0.08726646259971647 --plot
python ./scripts/calculate_bounds.py --dataset CIFAR10 --image-number 100 --padding-value 0 --lipschitz-error 0.05 --max-bab-iter 5000 --num-jobs=-1 --transformation ROTATE --lower 0 --upper 0.08726646259971647 --plot

# num_init_splits=1
python ./scripts/calculate_bounds.py --dataset MNIST --image-number 100 --padding-value 0 --lipschitz-error 0.05 --max-bab-iter 5000 --num-jobs=-1 --transformation ROTATE --lower 0 --upper 0.08726646259971647 --num-init-splits 1 --plot
python ./scripts/calculate_bounds.py --dataset CIFAR10 --image-number 100 --padding-value 0 --lipschitz-error 0.05 --max-bab-iter 5000 --num-jobs=-1 --transformation ROTATE --lower 0 --upper 0.08726646259971647 --num-init-splits 1 --plot
