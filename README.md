# Poker_Hand_Predict

UCI Poker Hand Dataset 扑克牌分类预测，实现99.96%的正确率

---

运行poker_with_UI,将显示pyqt制作的UI界面，提供可视化选择操作。

运行poker_without_UI,直接运行程序，选用Poker_Test列表中的值进行测试。

采用 DNN 算法、Tensorflow 深度学习框架， 对Flush, Straight flush, Royal flush情况另加处理。

具体技术原理及实现效果请看readme.pdf

---

运行结果如下

QT UI界面：

![image](https://github.com/HuangBingjian/Poker_Hand_Predict/blob/master/result/UI.png)

损失loss与训练步数的关系：

![image](https://github.com/HuangBingjian/Poker_Hand_Predict/blob/master/result/loss.png)

![image](https://github.com/HuangBingjian/Poker_Hand_Predict/blob/master/result/train_loss.png)

评估结果：

![image](https://github.com/HuangBingjian/Poker_Hand_Predict/blob/master/result/eval_result.png)

测试结果：

![image](https://github.com/HuangBingjian/Poker_Hand_Predict/blob/master/result/test_result.png)
