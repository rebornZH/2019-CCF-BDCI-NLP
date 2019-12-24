## 简要介绍
我是竞赛选手rebornZH，很高兴在2019 CCF BDCI大赛中拿下3个NLP竞赛TOP，在这里我**主要分享金融信息负面及主体判定赛题的解决方案和自己对另外两个NLP赛题的一些解题思路**，希望能与大家一起学习，一起进步，同时如果有NLP方向的朋友愿意和我一起交流学习NLP技术我也是非常欢迎的，可以加我QQ联系，后面有联系方式。最后希望大家多给下star，毕竟整理这些也挺花时间的，在这里先谢谢大家了。
## 比赛链接
* 金融信息负面及主体判定：https://www.datafountain.cn/competitions/353
* “技术需求”与“技术成果”项目之间关联度计算模型：https://www.datafountain.cn/competitions/359
* 互联网金融新实体发现：https://www.datafountain.cn/competitions/361
## 特别说明
* 上面的开源代码是在我最终版代码上修改而来的baseline，因为有些代码涉及到我队友，不方便全部开源，因此我特地整理一个比较好的基线给大家作为参考，这个基线的分数应该是该赛题开源的基线中分数最好的，并且距离最终版代码差别不是特别大。
* GPU最低要求：2080Ti 11G
## 金融信息负面及主体判定赛题—解决方案
![团队名](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/1.png)
* 团队名

![赛题介绍](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/9.png)
* 赛题介绍

![评价指标](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/10.png)
* 评价指标

![数据分析](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/2.png)
* 这一页主要是为了说明本赛题数据长度偏长

![联合训练](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/3.png)
* 这一页主要介绍方案的训练方式，我们采用end2end的方式，进行联合训练，用一个模型预测两个子任务

![传统深度学习方法](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/4.png)
* 这一页主要介绍之前尝试的传统深度学习方法，不过本赛题数据量有限，传统网络没有预训练权重，从头开始训练的话学习程度不够，效果较差，因此选择放弃

![BERT分类](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/5.png)
* 这一页主要介绍使用BERT分类的思想建模，关键是输入的设置，这也是本赛题上分的主要关键点之一，我们把输入设置为key_entity+分隔符+entity+分隔符+title+分隔符+text，其中entity加入到输入里面可以使得模型学习到很多规则的知识，比如最开始有人提到的预测出两个实体，一个实体包含了另一个实体，那么包含实体属于负面实体，被包含实体不属于负面实体。

![BERT匹配](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/6.png)
* 这一页主要介绍使用BERT匹配的思想建模，其他的和上一页ppt一样，这里不再叙述

![网络优化](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/7.png)
* 这一页主要介绍我们在优化器方面的尝试，RAdam效果最好

![数据增强](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/8.png)
* 这一页主要是通过数据增强的方法来提高数据利用率，前面介绍本赛题数据长度偏长主要就是为了体现这里，不过其实我感觉这个方案不是特别好，总的说来它的可用性很小，必须切分成固定长度，不太灵活，我们在“技术需求”与“技术成果”项目之间关联度计算模型赛题使用的基于滑窗的数据增强方法个人感觉要好一些。
## “技术需求”与“技术成果”项目之间关联度计算模型—个人思路
该赛题主要是匹配技术成果和技术需求，进行4分类。其实这个赛题的数据不是特别好，竟然还有相同样本不同标签的数据，这给我们的模型带来了很大的干扰，导致排行榜不稳定，不过我们团队的最终成绩是A榜第一、B榜第二，相信模型鲁棒性还是可以的。

![排行榜(https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/11.png)

这个赛题主要是两个输入长度的选择，我相信很多选手会把两个输入长度设置为一样长，其实我们发现这样效果不好，可以通过线下验证去调整每个输入的长度，选择合适的长度来输入，效果会提升很多，并且本赛题数据比较脏，需要做一下数据清洗（这也是我第一次在数据清洗后用BERT效果要好），还有一点就是本赛题的数据长度也是很长的，因此需要通过数据增强的方式来充分利用数据，而在这里我们就采用了滑窗的方法进行数据增强，如下图所示：

![数据增强](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/12.png)

## 互联网金融新实体发现—个人思路
这个赛题是一个序列标注问题，需要从提供的金融文本中识别出现的未知金融实体，包括金融平台名、企业名、项目名称及产品名称。首先我们需要搭建一个序列标注模型，BERT+CRF足矣，关键是这题数据文本也很长，有一些实体的位置在预训练模型最大长度512的后面，如果直接使用前512字来预测，那么肯定有些实体是无法识别的，因此我们需要把一篇文本切分成多个连续的子文本，每个子文本长度为512，然后把每个子文本进行训练并预测，把得到的所有实体汇总起来去重，剩下的就是最终预测出来的实体。最后就是模型融合加后处理，这里不再叙述，直接看下面的图。

![模型融合](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/13.png)

![规则后处理](https://github.com/rebornZH/2019-CCF-BDCI-NLP/blob/master/picture/14.png)

## 感想
可能通过这些介绍感觉工作量不是特别多，其实我还有很多尝试没有在这里一一说明，我主要是分享一些关键的地方来供大家学习交流。其实竞赛之路非常苦，从今年开始打NLP竞赛以来，一共获得了7个NLP竞赛TOP，非常累，也非常满足，不过中间确实也尝到了很多心酸和泪水，但是成长的道路就是这样，哪有什么一帆风顺，只有不断的乘风破浪，披荆斩棘，你才可能继续前进。

## Concat
QQ:1051296550

## 特别鸣谢
* https://github.com/guoday/CCF-BDCI-Sentiment-Analysis-Baseline




