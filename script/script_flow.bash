
dataset="GTEA61" 
trainDatasetDir="/content/GTEA61"
valDatasetDir="/content/GTEA61"
outDir="select the correct output folder"
seqLen=5 
trainBatchSize=32 
numEpochs=750 
lr=1e-2 
decayRate=0.1 
memSize=512
flag="flow"


main_run="/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/final_Project/main_run_self_supervised.py"

python $main_run --dataset $dataset --stage $stage --trainDatasetDir $trainDatasetDir --valDatasetDir $valDatasetDir --outDir $outDir --seqLen $seqLen --trainBatchSize $trainBatchSize --numEpochs $numEpochs --lr $lr -stepSize 150 300 500  --decayRate $decayRate --memSize $memSize --flag $flag

