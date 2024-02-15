#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

b=("2" "3" "4" "5" "6" "7" "8" "9" "10")
z_dim=("10" "30")
lr=("1e-4" "1e-3")
batch_size=("128" "64" "32")



for b in ${b[@]}
do
  for z_dim in ${z_dim[@]}
  do
    for lr in ${lr[@]}
    do
      for batch_size in ${batch_size[@]}
      do
        python egg_vae_trainer.py --b $b --z_dim $z_dim --lr $lr --batch_size $batch_size
      done
    done  
  done
done