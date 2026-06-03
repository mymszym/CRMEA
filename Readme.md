# CRMEA

1. **Datasets**: [DBP15K](https://github.com/nju-websoft/JAPE) and [OpenEA](https://github.com/nju-websoft/OpenEA)

2. **Data Processing**:
```bash
python ent_triple_generation.py -d=dze
```

3. **Train**:
```bash
python overall_process.py -d=dze -p=labse -l=qwen -m=train -r=0,1,2,3,4,5,6,7
```

