# RNN-SM: Fast Steganalysis of VoIP Streams Using Recurrent Neural Network


[[ paper (IEEE) ](http://ieeexplore.ieee.org/document/8292900/)] 
[[ paper (pdf) ](http://www.andrew.cmu.edu/user/zinanl/publications/rnn-sm.pdf)] 
[[ code ](https://github.com/fjxmlzn/RNN-SM#steganalysis-algorithms)] 
[[ dataset ](https://github.com/fjxmlzn/RNN-SM#steganalysis-speech-dataset)] 
[[ website ](https://github.com/fjxmlzn/RNN-SM)]

**Authors:** 
[Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/) ([Carnegie Mellon University](https://www.cmu.edu/)), 
[Yongfeng Huang](http://www.tsinghua.edu.cn/publish/ee/4157/2010/20101217182714916942750/20101217182714916942750_.html) ([Tsinghua University](http://www.tsinghua.edu.cn/publish/newthuen/index.html)), 
Jilong Wang ([Tsinghua University](http://www.tsinghua.edu.cn/publish/newthuen/index.html))

**Abstract:** Quantization index modulation (QIM) steganography makes it possible to hide secret information in voice-over IP (VoIP) streams, which could be utilized by unauthorized entities to set up covert channels for malicious purposes. Detecting short QIM steganography samples, as is required by real circumstances, remains an unsolved challenge. In this paper, we propose an effective online steganalysis method to detect QIM steganography. We find four strong codeword correlation patterns in VoIP streams, which will be distorted after embedding with hidden data. To extract those correlation features, we propose the codeword correlation model, which is based on recurrent neural network (RNN). Furthermore, we propose the feature classification model to classify those correlation features into cover speech and stego speech categories. The whole RNN-based steganalysis model (RNN-SM) is trained in a supervised learning framework. Experiments show that on full embedding rate samples, RNN-SM is of high detection accuracy, which remains over 90% even when the sample is as short as 0.1 s, and is significantly higher than other state-of-the-art methods. For the challenging task of conducting steganalysis towards low embedding rate samples, RNN-SM also achieves a high accuracy. The average testing time for each sample is below 0.15% of sample length. These clues show that RNN-SM meets the short sample detection demand and is a state-of-the-art algorithm for online VoIP steganalysis.

## Cite it 

This paper was published as a journal paper in IEEE Transactions on Information Forensics and Security.

```bash
@ARTICLE{lin2018rnn, 
	author={Lin, Zinan and Huang, Yongfeng and Wang, Jilong}, 
	journal={IEEE Transactions on Information Forensics and Security}, 
	title={RNN-SM: Fast Steganalysis of VoIP Streams Using Recurrent Neural Network}, 
	year={2018}, 
	volume={13}, 
	number={7}, 
	pages={1854-1868},
	doi={10.1109/TIFS.2018.2806741}, 
	ISSN={1556-6013},
	month={July}
}
```

## Steganalysis speech dataset

* Chinese speech

  [Chinese.tar.gz](https://drive.google.com/file/d/1LF2dAXHkd8TmzaDnTg0Zmbs7xVdSovMH/view?usp=sharing): 160 pieces of speech in wav format.

* English speech

  [English.tar.gz](https://drive.google.com/file/d/1Uy7WyEg3y-hvefUczo_6gFyyeeTC6ohg/view?usp=sharing): 160 pieces of speech in wav format.
  
## Steganalysis algorithms

* RNN-SM

  [algorithms/pruned_RNN_SM.py](https://github.com/fjxmlzn/RNN-SM/blob/master/algorithms/pruned_RNN_SM.py): our proposed pruned RNN-SM algorithms.

  [algorithms/full_RNN_SM.py](https://github.com/fjxmlzn/RNN-SM/blob/master/algorithms/full_RNN_SM.py): our proposed full RNN-SM algorithms.

* IDC

  [algorithms/IDC.py](https://github.com/fjxmlzn/RNN-SM/blob/master/algorithms/IDC.py): our implementation of [Detection of quantization index modulation steganography in G. 723.1 bit stream based on quantization index sequence analysis](https://link.springer.com/article/10.1631%2Fjzus.C1100374?LI=true)

* SS-QCCN

  [algorithms/SS_QCCN.py](https://github.com/fjxmlzn/RNN-SM/blob/master/algorithms/SS_QCCN.py): our implementation of [Steganalysis of QIM steganography in low-bit-rate speech signals](http://ieeexplore.ieee.org/abstract/document/7867798/)
