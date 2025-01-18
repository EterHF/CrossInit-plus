for (( i=0; i<8 i++ )); do
    start=$((i * 25))            
    end=$(((i + 1) * 25))        
    bash train_celebhq.sh "$start" "$end" "$i" &  
done

wait    