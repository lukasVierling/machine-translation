#!/bin/bash
echo "Source Running words:"
SourceWordCount=$(wc $1 | awk '{print $2}')
echo $SourceWordCount
echo "Target Running words:"
TargetWordCount=$(wc $2 | awk '{print $2}')
echo $TargetWordCount

#Replace Space with new line
#sort output, filter repeated words with unique
echo "Source Unique Words:"
tr '[:space:]' '\n' < $1 | sort | uniq | wc -l | awk '{print $1}'
echo "Target Unique Words:"
tr '[:space:]' '\n' < $2 | sort | uniq | wc -l | awk '{print $1}'

#delete all cahracters that are not a dot
#count dots beacuse numb dots = numb sentences
echo "Source Average Sentence Length"
#SrcNmbrSntc=$(tr -cd '.' < $1 | wc -c)
SrcNmbrSntc=$(wc -l $1 | awk '{print $1}')
echo $(( $SourceWordCount/$SrcNmbrSntc ))
echo "Target Average Sentence Length"
#TrgtNmbrSntc=$(tr -cd '.' < $2 | wc -c)
TrgtNmbrSntc=$(wc -l $1| awk '{print $1}')
echo $(( $TargetWordCount/$TrgtNmbrSntc ))
