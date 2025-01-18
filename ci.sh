for (( i=0; i<2; i++ )); do
    start=$((i * 25))            # 计算开始的文件夹下标
    end=$(((i + 1) * 25))        # 计算结束的文件夹下标
    bash run_ci.sh "$start" "$end" "$i" &  # 传递参数给 run_ci.sh
done

wait    