#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

batch_size=("32")
in_w=("256")
lr=("0.0005" "0.0001" "0.00005" "0.00001")
b=("0" "0.001" "0.0001" "0.00001")
linear_bn=("256" "512" "1024" "4096" "8192")
seed=("0")
scheduler=("step" "exp")
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
            for n in ${scheduler[@]}
            do
                python conv_vae.py --batch_size $i --in_w $j --lr $k --b $l --linear_bn $m --scheduler $n
            done
        done
      done
    done
  done
done
