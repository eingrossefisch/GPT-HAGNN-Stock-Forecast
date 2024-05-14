# HAN-Stock-Forecast

## 项目原理
在传统股市预测中，经常只关注单一股票的时序预测（ARIMA、LSTM、GRU），忽略了市场作为一个复杂网络，各市场主体存在的相互关联。我们希望构建一张金融关系图(Fin-Graph)反映市场主体之间的相互关联及其受新闻的冲击，同时我们也希望每个公司的股价和新闻的冲击能够互相影响彼此，在Fin-Graph像“水波”一样传播。  
本项目是基于HAGNN(Heterogeneous Attention Graph Neural Network, 可简写为HAN)是一种基于异构图神经网络进行股价时序预测的模型。在本模型中，我建立了一个刻画各公司之间相关关系的Fin-Graph，公司结点是异构的，包括股价嵌入、新闻和事件嵌入。同时，使用图神经网络的层次结构反映时序关系，每一层进行的aggregation可以看作是股市信息的一次传播。信息传播造成的影响通过连接公司结点的边权重来衡量。使用ChatGPT3.5将自然语言的新闻转化为影响公司的评分，归一化到[-1,1]后作为新闻和事件嵌入。在训练过程中，使用注意力机制学习边权重，以评估公司之间的相互关系的强弱。例如，A公司到B公司+1的边权重反映A上涨会造成B比例为1的上涨，而-2的边权重反映A上涨会造成B比例为2的下跌。  
金融关系图中包括了15家上证50指数非金融公司，每家公司都分配了一个公司结点，上面连接有股价结点以及新闻和事件结点。公司结点之间的影响是双向的，但是新闻结点对公司结点的影响是单向的。在每一层，所有结点的信息按照给定的方向进行扩散。在每一层都为新闻和事件节点更新成当天的新闻和事件嵌入。在实际训练中，我们考虑3天的时间窗口，即信息在Fin-Graph内传播三天。例如，假设模型边权重已经训练成熟，则在第一天发生新闻“高通断供小米”后，ChatGPT3.5会将此新闻识别为对小米-1的重创评分，在第二天，模型就会下调小米的股价，随后在第三天，模型会下调所有与小米股价正相关公司（如小米的供货商、小米的客户）的股价，并上调小米竞争对手的股价。但是实际情况中，预测的股价是所有公司共同作用的。  

## 结果分析
1. 如果将模型简化为二分类模型，最终的Macro-F1指数可以达到0.475，显著优于ARIMA, LSTM, GAT等模型。  
2. 模型只包含了242个交易日（划分80%作为训练集）的共4174条事件信息，最终依然给出了优秀的结果，证明了模型在小样本情况下的准确性。  
3. 在收益率方面表现优秀，面对2023年11月12日大盘从3000到2900点的下挫，使用基于本模型的投资策略只有1%的损失。  

## 代码使用
本项目是基于Python 3.11的，具体的环境要求可以参看"envs.txt"。您可以使用以下命令来配置环境：
```
conda create -n han python==3.11
pip install -r envs.txt
```
配置完毕环境后，运行`main.py`即可。本程序并不包括归一化功能，所以请确保你使用的数据集是经过归一化的。经过归一化后，替代`fin_data_normalized.xlsx`和`news_data_normalized.xlsx`即可。

## 参考模型
本模型参考了Cheng Dawei等老师发表在Pattern Recognition上的文献Financial Time Series Forecasting with Multi-modality Graph Neural Network。我真的很希望GitHub有引用功能。
```
@article{cheng2022financial,
  title={Financial time series forecasting with multi-modality graph neural network},
  author={Cheng, Dawei and Yang, Fangzhou and Xiang, Sheng and Liu, Jin},
  journal={Pattern Recognition},
  volume={121},
  pages={108218},
  year={2022},
  publisher={Elsevier}
}
```
