# RNN-SM: Fast Steganalysis of VoIP Streams Using Recurrent Neural Network

Authors: [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/) ([Carnegie Mellon University](https://www.cmu.edu/)), [Yongfeng Huang](http://www.tsinghua.edu.cn/publish/ee/4157/2010/20101217182714916942750/20101217182714916942750_.html) ([Tsinghua University](http://www.tsinghua.edu.cn/publish/newthuen/index.html))

Website: https://github.com/fjxmlzn/RNN-SM

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
