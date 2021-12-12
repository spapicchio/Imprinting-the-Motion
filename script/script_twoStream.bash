dataset="GTEA61"
trainDatasetDir="/content/GTEA61"
valDatasetDir="/content/GTEA61"

seqLen=7 
trainBatchSize=32 
numEpochs=250 
lr=1e-2
decayRate=0.99 
memSize=512
flag="jointStream"

main_run="/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/final_Project/main_run_self_supervised.py"
outDir="select the correct output folder"
flowModel="select the best_model_state_dict_flow_split2.pth"
rgbModel="select the best_model_state_dict_rgb_split2.pth"


python $main_run --dataset $dataset --flowModel $flowModel --rgbModel $rgbModel --trainDatasetDir $trainDatasetDir --valDatasetDir $valDatasetDir --outDir $outDir --seqLen $seqLen --trainBatchSize $trainBatchSize --numEpochs $numEpochs --lr $lr -stepSize 1  --decayRate $decayRate --memSize $memSize --flag $flag
