# Curriculum KD

Command to Train using CKD on BUS Dataset <br>

```
python main.py --data bus --model_name CKD --model_path [Path of saving final model] --teacher_path [Path to pretrained teacher model]
```

Command to Test using CKD on BUS Dataset <br>

```
python test.py --data bus --model_name CKD --model_path [Path to model to be evaluated]
```