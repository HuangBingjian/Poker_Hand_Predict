'''
作者：黄炳坚

运行poker_with_UI,将显示pyqt制作的UI界面，提供可视化选择操作。
运行poker_without_UI,直接运行程序，选用Poker_Test列表中的值进行测试。
'''

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import tensorflow as tf
from poker_without_UI import poker_predict,Poker_Test

class show_window(QWidget):
    def __init__(self):
        super(show_window, self).__init__()
        self.initUI()

    def initUI(self):
        self.setFixedSize(480,550)
        self.setWindowTitle('扑克牌分类')

        titleLabel = QLabel("机器学习期末项目——扑克牌分类",self)
        titleLabel.setFont(QFont("Roman times",15 ,QFont.Bold))
        titleLabel.move(50, 50)
        authorLabel = QLabel("作者：黄炳坚",self)
        authorLabel.move(350, 100)

        rank = ['   A','   2','   3','   4','   5','   6','   7','   8','   9','   10','   J','   Q','   K']
        suit = ['   红心(1)','   黑桃(2)','   方片(3)','   梅花(4)']
        poker_Label = QLabel('扑克牌',self)
        poker_Label.move(50,150)
        card1_Label = QLabel("第一张：",self)
        card1_Label.move(50,210)
        self.rank1_Combo = QComboBox(self)
        self.rank1_Combo.move(300,200)
        self.rank1_Combo.addItems(rank)
        self.rank1_Combo.setFixedSize(80,30)
        self.suit1_Combo = QComboBox(self)
        self.suit1_Combo.move(150,200)
        self.suit1_Combo.addItems(suit)
        self.suit1_Combo.setFixedSize(120,30)
        card2_Label = QLabel("第二张：",self)
        card2_Label.move(50,260)
        self.rank2_Combo = QComboBox(self)
        self.rank2_Combo.move(300,250)
        self.rank2_Combo.addItems(rank)
        self.rank2_Combo.setFixedSize(80,30)
        self.suit2_Combo = QComboBox(self)
        self.suit2_Combo.move(150,250)
        self.suit2_Combo.addItems(suit)
        self.suit2_Combo.setFixedSize(120,30)
        card3_Label = QLabel("第三张：",self)
        card3_Label.move(50,310)
        self.rank3_Combo = QComboBox(self)
        self.rank3_Combo.move(300,300)
        self.rank3_Combo.addItems(rank)
        self.rank3_Combo.setFixedSize(80,30)
        self.suit3_Combo = QComboBox(self)
        self.suit3_Combo.move(150,300)
        self.suit3_Combo.addItems(suit)
        self.suit3_Combo.setFixedSize(120,30)
        card4_Label = QLabel("第四张：",self)
        card4_Label.move(50,360)
        self.rank4_Combo = QComboBox(self)
        self.rank4_Combo.move(300,350)
        self.rank4_Combo.addItems(rank)
        self.rank4_Combo.setFixedSize(80,30)
        self.suit4_Combo = QComboBox(self)
        self.suit4_Combo.move(150,350)
        self.suit4_Combo.addItems(suit)
        self.suit4_Combo.setFixedSize(120,30)
        card5_Label = QLabel("第五张：",self)
        card5_Label.move(50,410)
        self.rank5_Combo = QComboBox(self)
        self.rank5_Combo.move(300,400)
        self.rank5_Combo.addItems(rank)
        self.rank5_Combo.setFixedSize(80,30)
        self.suit5_Combo = QComboBox(self)
        self.suit5_Combo.move(150,400)
        self.suit5_Combo.addItems(suit)
        self.suit5_Combo.setFixedSize(120,30)

        poker_Btn = QPushButton('确定',self)
        poker_Btn.move(150,480)
        poker_Btn.setFixedSize(230,40)
        poker_Btn.clicked.connect(self.poker_classifier)

        self.show()

    def poker_classifier(self):
        self.hide()
        Poker_Test[0] = self.suit1_Combo.currentIndex()+1
        Poker_Test[1] = self.rank1_Combo.currentIndex()+1
        Poker_Test[2] = self.suit2_Combo.currentIndex()+1
        Poker_Test[3] = self.rank2_Combo.currentIndex()+1
        Poker_Test[4] = self.suit3_Combo.currentIndex()+1
        Poker_Test[5] = self.rank3_Combo.currentIndex()+1
        Poker_Test[6] = self.suit4_Combo.currentIndex()+1
        Poker_Test[7] = self.rank4_Combo.currentIndex()+1
        Poker_Test[8] = self.suit5_Combo.currentIndex()+1
        Poker_Test[9] = self.rank5_Combo.currentIndex()+1
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run(poker_predict)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = show_window()
    sys.exit(app.exec_())