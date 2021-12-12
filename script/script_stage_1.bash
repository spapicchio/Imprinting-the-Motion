dataset="GTEA61"
stage=1 
trainDatasetDir="/content/GTEA61"
valDatasetDir="/content/GTEA61"
seqLen=7 
trainBatchSize=32 
numEpochs=300 
lr=1e-3 
decayRate=0.1 
memSize=512

main_run="/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/final_Project/main_run_self_supervised.py"
outDir="correct output folder"
flag="select correct flag"

python $main_run --dataset $dataset --stage $stage  --trainDatasetDir $trainDatasetDir --valDatasetDir $valDatasetDir --outDir $outDir --seqLen $seqLen --trainBatchSize $trainBatchSize --numEpochs $numEpochs --lr $lr --stepSize 25 75 150 --decayRate $decayRate --memSize $memSize --flag $flag