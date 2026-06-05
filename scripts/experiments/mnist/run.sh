python ./scripts/calculate_bounds.py --dataset MNIST --image-number 99 --padding-value 0 --lipschitz-error 0.01 --max-bab-iter 5000 --num-jobs=-1 --transformation H_ROLL --lower 0 --upper 0.08726646259971647
python ./scripts/calculate_bounds.py --dataset MNIST --image-number 99 --padding-value 0 --lipschitz-error 0.01 --max-bab-iter 5000 --num-jobs=-1 --transformation H_PITCH --lower 0 --upper 0.08726646259971647
python ./scripts/calculate_bounds.py --dataset MNIST --image-number 99 --padding-value 0 --lipschitz-error 0.01 --max-bab-iter 5000 --num-jobs=-1 --transformation H_YAW --lower 0 --upper 0.08726646259971647

python ./scripts/calculate_bounds.py --dataset MNIST --image-number 99 --padding-value 0 --lipschitz-error 0.01 --max-bab-iter 5000 --num-jobs=-1 --transformation H_X --lower 0 --upper 1
python ./scripts/calculate_bounds.py --dataset MNIST --image-number 99 --padding-value 0 --lipschitz-error 0.01 --max-bab-iter 5000 --num-jobs=-1 --transformation H_Y --lower 0 --upper 1
python ./scripts/calculate_bounds.py --dataset MNIST --image-number 99 --padding-value 0 --lipschitz-error 0.01 --max-bab-iter 5000 --num-jobs=-1 --transformation H_Z --lower 0 --upper 1
