python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCBL
mv distill_run/ distill_run_BCBL0/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCBL
mv distill_run/ distill_run_BCBL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCBL
mv distill_run/ distill_run_BCBL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCBL
mv distill_run/ distill_run_SCBL0/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCBL
mv distill_run/ distill_run_SCBL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCBL
mv distill_run/ distill_run_SCBL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCSL
mv distill_run/ distill_run_BCSL0/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCSL
mv distill_run/ distill_run_BCSL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type BCSL
mv distill_run/ distill_run_BCSL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCSL
mv distill_run/ distill_run_SCSL0/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCSL
mv distill_run/ distill_run_SCSL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --student_model_type SCSL
mv distill_run/ distill_run_SCSL2/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --teacher_led
mv distill_run/ distill_run_TL0/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --teacher_led
mv distill_run/ distill_run_TL1/
python3 distill.py --teacher_path pong_run_steven/pong_expert.zip --teacher_led
mv distill_run/ distill_run_TL2/
python3 distill.py --teacher_path master.zip
mv distill_run/ distill_run_master0/
python3 distill.py --teacher_path master.zip
mv distill_run/ distill_run_master1/
python3 distill.py --teacher_path master.zip
mv distill_run/ distill_run_master2/