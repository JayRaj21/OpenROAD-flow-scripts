python3 util/ml/congestion/inference/visualize_thermal.py \
    --data-dir util/ml/congestion/data \
    --checkpoint util/ml/congestion/checkpoints/thermal_best.pt \
    --out thermal_report.html \
    2>&1 | tee util/ml/congestion/visualize_thermal.log
