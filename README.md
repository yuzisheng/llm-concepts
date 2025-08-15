一口气通关大模型的100个概念

1. 人工智能（Artificial Intelligence, AI）  
一句话解释：让机器像人一样“思考”和“干活”的技术总称。  

2. 机器学习（Machine Learning, ML）  
一句话解释：让电脑通过数据自己总结规律、做出决策的算法集合。  

3. 深度学习（Deep Learning, DL）  
一句话解释：用多层“神经网络”来发现数据中复杂模式的机器学习分支。  

4. 神经网络（Neural Network）  
一句话解释：模仿人脑神经元的计算结构，层层传递信号并自动提取特征。  

5. 参数（Parameters）  
一句话解释：模型中可调的“旋钮”，决定它如何理解并生成答案。  

6. 权重（Weights）  
一句话解释：神经网络里连接强度的大小，数值越大表示越重要。  

7. 偏置（Bias）  
一句话解释：给神经元一个“基础倾向”，让模型更容易拟合数据。  

8. 激活函数（Activation Function）  
一句话解释：决定神经元是否“点火”的开关，给网络带来非线性能力。  

9. 损失函数（Loss Function）  
一句话解释：衡量模型输出与真实答案差距的“尺子”，越小越好。  

10. 梯度（Gradient）  
一句话解释：损失函数在某点的“坡度”，告诉我们该往哪个方向调参。  

11. 梯度下降（Gradient Descent）  
一句话解释：沿最陡的下坡路一点点挪动参数，直到损失最小。  

12. 反向传播（Backpropagation）  
一句话解释：把误差从后往前逐层“倒推”，算出每层参数该怎么改。  

13. 预训练（Pre-training）  
一句话解释：先用海量通用文本把模型“读胖”，再针对任务“减肥”微调。  

14. 微调（Fine-tuning）  
一句话解释：在预训练模型上加少量特定任务数据，做轻微调参以专精某项工作。  

15. Transformer（Transformer）  
一句话解释：用“自注意力”同时看一整句话，彻底取代逐字循环的划时代架构。  

16. 注意力机制（Attention Mechanism）  
一句话解释：给句子里的每个词打分，决定“看”谁多一点、“看”谁少一点。  

17. 自注意力（Self-Attention）  
一句话解释：句子自己内部做注意力，让每个词都能结合上下文重新定义自己。  

18. 位置编码（Positional Encoding）  
一句话解释：告诉模型“我在第几位”，因为Transformer本身对顺序不敏感。  

19. 编码器（Encoder）  
一句话解释：把输入文本转成富含语义的高维向量表示的“翻译器”。  

20. 解码器（Decoder）  
一句话解释：根据Encoder的输出生成目标序列，比如把中文翻成英文。  

21. 编码器唯一架构（Encode-only）  
一句话解释：只保留Transformer的编码器，一次性读取整句并输出语义向量，典型代表BERT系列，擅长理解与分类。

22. 解码器唯一架构（Decode-only）  
一句话解释：只保留Transformer的解码器，自左向右逐token生成文本，典型代表GPT系列，擅长续写与对话。

23. 大语言模型（Large Language Model, LLM）  
一句话解释：参数量巨大、训练数据海量、能说话、能写作、能推理的语言模型。  

24. 参数规模（Parameter Scale）  
一句话解释：模型里可调权重的总个数，通常用B（十亿）作单位，越大越“聪明”。  

25. GPT（Generative Pre-trained Transformer）  
一句话解释：只用解码器、从左到右生成文字的大模型家族，擅长写作和对话。  

26. BERT（Bidirectional Encoder Representations from Transformers）  
一句话解释：只用编码器、双向看句子的大模型，擅长理解类任务如问答、分类。  

27. T5（Text-to-Text Transfer Transformer）
一句话解释：把“所有任务都变成文字到文字”的编码器-解码器大模型。

28. 词向量（Word Embedding）  
一句话解释：把单词变成可计算的数字坐标，语义相近的词坐得近。  

29. 词元（Token）  
一句话解释：模型实际读写的最小单元，可能是整词、子词或单个字符。  

30. 分词器（Tokenizer）  
一句话解释：把原始文本切成模型能读懂的最小单元（token）的“翻译官”，同时能把token再拼回原文，实现文字与数字的双向转换。  

31. 词表（Vocabulary）  
一句话解释：模型认识的所有token的“字典”，大小通常在几万到几十万。  

32. 嵌入层（Embedding Layer）  
一句话解释：把token编号映射成稠密向量的“查找表”，是语义旅程的起点。  

33. 多头注意力（Multi-Head Attention）  
一句话解释：把自注意力拆成多组“头”，每组学不同关系，像多镜头拍同一画面。  

34. 残差连接（Residual Connection）  
一句话解释：把输入直接加到输出，防止深层网络“信号衰减”，让梯度更顺畅。  

35. 层归一化（Layer Normalization, LayerNorm）  
一句话解释：对一层神经元的输出做“归一化”，训练更稳、收敛更快。  

36. 下一个词预测（Next Token Prediction）  
一句话解释：让模型根据已出现的所有上文，猜出接下来最可能出现的词（token）；通过不断把“猜对”作为奖励，模型在海量文本中自学语法、事实和推理能力。

37. 训练语料（Training Corpus）  
一句话解释：喂给模型的“教科书”，越多越广，模型知识面越宽。  

38. 数据清洗（Data Cleaning）  
一句话解释：去掉乱码、广告、重复等垃圾，让模型吃“干净饭”。  

39. 数据去重（Data Deduplication）  
一句话解释：删掉重复文本，防止模型“背熟”而“背傻”。  

40. 数据配比（Data Mixing Ratio）  
一句话解释：不同领域、语言、体裁按一定比例混合，避免模型偏科。  

41. 数据增强（Data Augmentation）  
一句话解释：用同义改写、回译等手法给数据“加量不加价”，提升泛化能力。  

42. 训练步数（Training Steps）  
一句话解释：模型看过并更新参数的总批次数，步数越多学得越久。  

43. 批次（Batch）  
一句话解释：一次同时喂给模型的一组样本，显卡吃得下就尽量大。  

44. 学习率（Learning Rate）  
一句话解释：每次调参的步子大小，太大翻车，太小磨蹭。  

45. 预热（Warm-up）  
一句话解释：训练初期先小步慢跑，防止一开始就猛冲导致发散。  

46. 学习率衰减（Learning Rate Decay）  
一句话解释：训练后期逐步减小学习率，像踩刹车，稳稳收拢到最优点。  

47. 混合精度训练（Mixed Precision Training）  
一句话解释：同时用16位和32位浮点数，省显存、加速还能保精度。  

48. 梯度累积（Gradient Accumulation）  
一句话解释：把多个小批的梯度攒起来再更新，模拟大批次效果。  

49. 模型并行（Model Parallelism）  
一句话解释：把网络的不同层放在不同GPU，解决“一张卡装不下”的问题。  

50. 数据并行（Data Parallelism）  
一句话解释：把同一份模型复制到多张卡，各自吃不同数据，再同步梯度。  

51. ZeRO / DeepSpeed  
一句话解释：微软的显存优化技术，把参数、梯度、优化器状态拆得更碎，省显存。  

52. 检查点（Checkpoint）  
一句话解释：训练过程中的“存档点”，可断电续练，也可拿来做微调。  

53. 早停（Early Stopping）  
一句话解释：验证集表现不再提升就停止训练，防止过拟合浪费时间。  

54. 过拟合（Overfitting）  
一句话解释：模型把训练题背得滚瓜烂熟，但一考新题就懵。  

55. 正则化（Regularization）  
一句话解释：给损失加“惩罚项”，逼模型别死记，提高泛化。  

56. 随机失活（Dropout）  
一句话解释：训练时随机“掐掉”部分神经元，逼网络用冗余路径，防过拟合。  

57. 泛化能力（Generalization）  
一句话解释：模型在新数据上表现依旧良好的“抗考”能力。  

58. 评估指标（Evaluation Metrics）  
一句话解释：用来量化模型好坏的尺子，如准确率、BLEU、ROUGE、困惑度。  

59. 困惑度（Perplexity）  
一句话解释：语言模型看句子时“惊讶程度”，越低越自信。  

60. 人工评测（Human Evaluation）  
一句话解释：让人类打分，最直观也最贵，常用来验证自动指标靠不靠谱。  

61. 提示词（Prompt）  
一句话解释：给模型的“任务说明书”，一段话决定它回答的风格和内容。  

62. 提示词工程（Prompt Engineering）  
一句话解释：把说明书写得又精又巧，让模型少犯错、多出彩。  

63. 零样本学习（Zero-shot Learning）  
一句话解释：不给示例，只靠任务描述就让模型直接上阵。  

63. 少样本学习（Few-shot Learning）  
一句话解释：给模型看几个示例就能举一反三，无需大量再训练。  

65. 上下文学习（In-context Learning）  
一句话解释：在对话窗口里即时演示，模型当场学会新任务。  

66. 思维链（Chain-of-Thought, CoT）  
一句话解释：让模型把思考过程一步步说出来，提高复杂推理准确率。  

67. 温度（Temperature）  
一句话解释：控制生成随机性的旋钮，越高越有创意，越低越保守。  

68. Top-k 采样（Top-k Sampling）  
一句话解释：每一步只从概率最高的k个词中选，平衡多样性与可读性。  

69. Top-p 采样（Top-p / Nucleus Sampling）  
一句话解释：按累计概率动态圈词池，更灵活地控制生成长尾词。  

70. 束搜索（Beam Search）  
一句话解释：每一步保留多条“候选句子”，最后挑总分最高的那条。  

71. 长度惩罚（Length Penalty）  
一句话解释：给长句打折扣，防止Beam Search生成冗长废话。  

72. 幻觉（Hallucination）  
一句话解释：模型一本正经地编出不存在的“事实”。  

73. 对齐（Alignment）  
一句话解释：让模型的价值观、目标与人类一致，不跑偏、不胡来。  

74. 人类反馈强化学习（RLHF, Reinforcement Learning from Human Feedback）  
一句话解释：用“人点赞”当奖励信号，强化模型输出更合人意的答案。  

75. 近端策略优化（PPO, Proximal Policy Optimization）  
一句话解释：一种高效的强化学习算法，常用于RLHF阶段。  

76. 奖励模型（Reward Model）  
一句话解释：先训练一个小裁判，给回答打分，再指导大模型改进。  

77. 知识蒸馏（Knowledge Distillation）  
一句话解释：让大模型当老师，把知识“蒸”到小模型，省算力。  

78. 量化（Quantization）  
一句话解释：把32位权重压缩到8位甚至4位，模型瘦身跑更快。  

79. 剪枝（Pruning）  
一句话解释：砍掉不重要的权重或神经元，让模型轻装上阵。  

80. 稀疏化（Sparsification）  
一句话解释：让权重矩阵里大量变0，既省存储又省计算。  

81. KV-Cache  
一句话解释：把之前算过的键值存起来，避免重复计算，长文生成必备。  

82. PagedAttention
一句话解释：像操作系统分页一样管理KV-Cache，显存碎片几乎清零。  

83. FlashAttention
一句话解释：在GPU SRAM 里完成注意力计算，少搬数据多干活。  

84. 持续学习（Continual Learning）  
一句话解释：模型学会新知识不忘旧知识，像人终身学习。  

85. 灾难性遗忘（Catastrophic Forgetting）  
一句话解释：学新任务把旧任务忘光的尴尬现象。  

86. 参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）  
一句话解释：只动一小撮参数就能完成微调，LoRA、Adapter都属于它。  

87. 多模态大模型（Multimodal Large Model）  
一句话解释：既能看文字也能看图、听声音，跨感官理解世界。  

88. Vit（Vision Transformer）  
一句话解释：把图片切成小图块当“文字”，用 Transformer 看得懂图。  

89. CLIP（Contrastive Language–Image Pre-training）  
一句话解释：OpenAI 的图文对比学习模型，懂得“文字描述找图片”。  

90. 扩散模型（Diffusion Model）
一句话解释：通过“加噪再降噪”生成高质量图像的魔法。

91. AIGC（AI Generated Content）  
一句话解释：用AI自动生产文字、图片、音频、视频等一切内容。  

92. 本地部署（Local Deployment）  
一句话解释：把大模型搬进自己服务器或电脑，数据不出境，隐私更安心。  

93. 边缘计算（Edge Computing）  
一句话解释：把轻量化模型塞进手机、IoT设备，离线也能智能对话。

94. Function Call（函数调用）  
一句话解释：大模型在生成回答时主动调用预定义函数，把外部计算结果无缝拼回对话。 

95. MCP（Model Context Protocol，模型上下文协议）  
一句话解释：一种让大模型与外部工具/数据源实时交互的标准通信协议，保证上下文一致且安全。  

96. Agent（智能体）  
一句话解释：以大模型为“大脑”，结合工具链、记忆、规划，自主完成多步任务的系统。  

97. ReAct（Reason+Act）  
一句话解释：把推理（Thought）与行动（Action）交替进行，使 Agent 边想边干。  

98. RAG（Retrieval-Augmented Generation）  
一句话解释：先检索外部知识，再让大模型生成，既实时又准确。  

99. 向量数据库（Vector Database）  
一句话解释：把文本转成高维向量后做极速相似搜索，可为RAG提供记忆。  

100. Embedding 模型（Embedding Model）  
一句话解释：专门负责把句子/图片转成向量的“翻译官”，供检索或聚类使用。  
