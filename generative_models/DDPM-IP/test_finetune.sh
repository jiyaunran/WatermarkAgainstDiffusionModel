#!/bin/bash

i="0"
j="0"
TrainingFolder="/work/jyran2001208/data/finetune_test10000_start20000/"
CleanDataFolder="/work/jyran2001208/data/hidden/clean/fingerprinted_images/"
FingerprintLength="100"
ImageResolution="64"
PoisonNum="10000"
FinetuneStep="10000"
PretrainedModel="/work/jyran2001208/GenModel/DDPM-IP/ema_0.9999_040000.pt"
SampleNum=3000

TrainEnDecoder ( ) {
    cd /home/jyran2001208/model/promote_poison_hidden/phase1/ArtificialGANFingerprints/
    if [ "$3" == true ]
    then
        bash train.sh \
        -d "$CleanDataFolder" \
        -i "$ImageResolution" \
        -o "${TrainingFolder}" \
        -f "$FingerprintLength" \
        -b 64 \
        -u true \
        -r "${TrainingFolder}regression/Regressor.pt"
    else
        bash train.sh \
        -d "$CleanDataFolder" \
        -i "$ImageResolution" \
        -o "${TrainingFolder}" \
        -f "$FingerprintLength" \
        -b 64
    fi
}

Generate ( ) {
    cd /home/jyran2001208/model/promote_poison_hidden/phase1/ArtificialGANFingerprints/
    mkdir "${TrainingFolder}/embed_img/"
    rm -r "${TrainingFolder}/embed_img/*"
    ##use encoder generate poison generator
    
    bash embed.sh \
    -e "$EncoderPath" \
    -d "$CleanDataFolder" \
    -i "$ImageResolution" \
    -o "${TrainingFolder}embed_img/" \
    -f "$FingerprintLength" \
    -b 64 \
    -p "$PoisonNum"
}

SaveEnDecoderName ( ) {
    EncoderPath=`ls ${TrainingFolder}/checkpoints/*encoder.pth`
    DecoderPath=`ls ${TrainingFolder}/checkpoints/*decoder.pth`
    echo "Encoder${2} path: ${EncoderPath}" >> ${Logtxt}
    echo "Decoder${2} path: ${DecoderPath}" >> ${Logtxt}
}

TrainDDPM ( ) {
    cd /home/jyran2001208/model/promote_poison_hidden/phase2/DDPM-IP/scripts/
    next=$(( ${2} + 1 ))
    fine_step=5000
    prevModel=`ls ${TrainingFolder}model/${2}/ema*`
    bash train.sh \
    -d "${TrainingFolder}embed_img/fingerprinted_images/" \
    -i "$ImageResolution" \
    -b 32 \
    -s "${fine_step}" \
    -v "${TrainingFolder}model/${next}/" \
    -r "${prevModel}"
}

SampleDDPM ( ) {
    cd /home/jyran2001208/model/promote_poison_hidden/phase2/DDPM-IP/scripts/
    next=$(( ${2} + 1 ))
    prevModel=`ls ${TrainingFolder}model/${next}/ema*`
    bash sample.sh \
    -m "${prevModel}" \
    -i "$ImageResolution" \
    -b 256 \
    -n "$SampleNum" \
    -o "${TrainingFolder}model/${next}/"
}

DetectImg ( ) {
    cd /home/jyran2001208/model/promote_poison_hidden/phase1/ArtificialGANFingerprints/
    prevModel=`ls ${TrainingFolder}model/${2}/ema*`
    bash detect.sh \
    -a "$DecoderPath" \
    -d "${TrainingFolder}model/${2}/gen_img/" \
    -i "$ImageResolution" \
    -r "${TrainingFolder}${2}/regression/" \
    -f "$FingerprintLength" \
    -b 256 \
    -c false
}


UseReg=false

#train encoder
#TrainEnDecoder "$i" "$j" "$UseReg"
SaveEnDecoderName "$i" "$j"
#Generate "$i" "$j"
#
#for j in $(seq 0 6)
#do
#    TrainDDPM "$i" "$j"
#    SampleDDPM "$i" "$j"
#    next=$(( ${j} + 1 ))
#    npzfile=`ls ${TrainingFolder}model/${next}/*.npz`
#    mkdir "${TrainingFolder}model/${next}/gen_img"
#    python npz2png.py --input_path "$npzfile" --output_path "${TrainingFolder}model/${next}/"
#done

for j in $(seq 1 7)
do
    DetectImg "$i" "$j"
done