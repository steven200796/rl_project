python graph.py --title="Averaged Student-Led vs Teacher-Led Distillation (n=3)" --save figures/lead_comparison_avg evals/lead_comparison/teacher_led evals/lead_comparison/student_led
python graph.py --title="Teacher-Led Distillation Runs" --save figures/teach_led evals/lead_comparison/teacher_led/*
