python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCBL
mv distill_run/ distill_run_BCBL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCBL
mv distill_run/ distill_run_BCBL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCBL
mv distill_run/ distill_run_SCBL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCBL
mv distill_run/ distill_run_SCBL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCSL
mv distill_run/ distill_run_BCSL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCSL
mv distill_run/ distill_run_BCSL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCSL
mv distill_run/ distill_run_SCSL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCSL
mv distill_run/ distill_run_SCSL2/