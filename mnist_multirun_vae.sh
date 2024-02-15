#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

batch_size=("64")
in_w=("28")
lr=("0.001")
b=("2" "3" "4")
linear_bn=("10" "30")
seed=("0")
for i in ${batch_size[@]}
do
  for j in ${in_w[@]}
  do
    for k in ${lr[@]}
    do
      for l in ${b[@]}
      do
        for m in ${linear_bn[@]}
        do
            for n in ${seed[@]}
            do
                python conv_vae.py --batch_size $i --in_w $j --lr $k --b $l --linear_bn $m --seed $n --dataset MNIST
            done
        done
      done
    done
  done
done
