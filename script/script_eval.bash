eval_run="/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/final_Project/eval_self_model.py"

dataset="GTEA61"
datasetDir="/content/GTEA61"
memSize=512

modelStateDict="correct model"
flag="correct_flag"
seqLen=7 

python $eval_run --dataset $dataset --datasetDir $datasetDir --seqLen $seqLen --memSize $memSize --flag $flag --modelStateDict $modelStateDict
