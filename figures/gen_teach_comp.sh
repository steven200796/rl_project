python graph.py --title="Averaged Expert Teacher vs Master Teacher Distillation (n=3)" --save figures/teacher_comparison_avg evals/teacher_comparison/master_teacher evals/teacher_comparison/expert_teacher
python graph.py --title="Master Teacher Distillation Runs" --save figures/master_teacher evals/teacher_comparison/master_teacher/*
