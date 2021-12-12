dataset="GTEA61"
stage=2 
trainDatasetDir="/content/GTEA61"
valDatasetDir="/content/GTEA61"
seqLen=7 
trainBatchSize=32 
numEpochs=150 
lr=1e-4 
decayRate=0.1 
memSize=512

main_run="/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/final_Project/main_run_self_supervised.py"
stage1Dict="select the best model" 
outDir="select the correct output"
flag="select correct flag"

python $main_run --dataset $dataset --stage $stage --stage1Dict $stage1Dict --trainDatasetDir $trainDatasetDir --valDatasetDir $valDatasetDir --outDir $outDir --seqLen $seqLen --trainBatchSize $trainBatchSize --numEpochs $numEpochs --lr $lr --stepSize 25 75  --decayRate $decayRate --memSize $memSize --flag $flag

