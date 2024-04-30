python graph.py --title="Averaged Distillation Runs (n=3)" --save figures/distill_avg evals/distillation_runs/bcbl evals/distillation_runs/bcsl evals/distillation_runs/scbl evals/distillation_runs/scsl
python graph.py --title="BCBL Distillation Runs" --save figures/distill_bcbl evals/distillation_runs/bcbl/* 
python graph.py --title="BCSL Distillation Runs" --save figures/distill_bcsl evals/distillation_runs/bcsl/* 
python graph.py --title="SCBL Distillation Runs" --save figures/distill_scbl evals/distillation_runs/scbl/* 
python graph.py --title="SCSL Distillation Runs" --save figures/distill_scsl evals/distillation_runs/scsl/*
